import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import torch
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer, build_ACT_prediction_model_and_optimizer, build_ACT2_model_and_optimizer, build_MINE_model_and_optimizer
import IPython
e = IPython.embed

class Predict_Model(nn.Module):
    def __init__(self, args_override):
        super(Predict_Model, self).__init__()
        self.args = args_override
        self.model, self.optimizer = build_ACT_prediction_model_and_optimizer(args_override)
        self.num_queries = args_override['num_queries']
        self.decay_rate = args_override['decay_rate']
    def configure_optimizers(self):
        return self.optimizer

    def __call__(self, qpos, image, features=None, is_pad=None, is_training=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        # if features is not None: # training time
        # features : [batch_size, num_queries, feature_dim]
        # print(features.shape, " features\n")
        # print(is_pad.shape, " is_pad\n")
        features = features[:, :self.num_queries]
        is_pad = is_pad[:, :self.num_queries]
        features_hat = self.model(qpos, image)
        ### assume only one camera
        features = features.squeeze(2)
        #########################################
        # print(features.shape, " features\n")
        # print(features_hat.shape, " features_hat\n")
        loss_dict = dict()
        # all_l2_with_decay
        all_l2 = F.mse_loss(features, features_hat, reduction='none')
        ## add decay
        # decay_factor = self.decay_rate ** torch.arange(all_l2.shape[-1], device=all_l2.device)
        # all_l2_with_decay = all_l2 * decay_factor

        l2 = (all_l2 * ~is_pad.unsqueeze(-1)).mean()
        ### end of add decay
        loss_dict['l2'] = l2
        loss_dict['loss'] = loss_dict['l2']
        return all_l2,loss_dict
        # else: # inference time
        #     a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
        #     return a_hat
        # return latent
    
    def get_features(self, qpos, image):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        features_hat = self.model(qpos, image)
        return features_hat

class VAEPredictionModel(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.decay_rate = args_override['decay_rate']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, future_features=None, is_pad=None, is_training=True):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        old_image = image
        image = normalize(image)
        if is_training: # training time
            future_features = future_features[:, :self.model.num_queries]
            if len(future_features.shape) == 4:
                future_features = future_features.squeeze(2)
            is_pad = is_pad[:, :self.model.num_queries]

            feature_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, future_features, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            # all_l1_with_decay
            all_l2 = F.mse_loss(future_features, feature_hat, reduction='none')
            ## add decay
            # decay_factor = self.decay_rate ** torch.arange(all_l1.shape[-1], device=all_l1.device)
            # if predict_model is not None:
            #     with torch.inference_mode():
            #         # predict_model(qpos_data, image_data, image_rep, is_pad)
            #         loss, L = predict_model(qpos, old_image,image_features,is_pad)
            #         loss = loss.sum(-1)
            #         loss = (~is_pad) * loss
            #         print(L['l2'], " l2\n")
            #         print(loss.shape, " loss\n")
            #         print(loss,end=" loss\n")
            #         exit(0)


            # else:
            #     all_l1_with_decay = all_l1 * decay_factor


            l2 = (all_l2 * ~is_pad.unsqueeze(-1)).mean()
            ### end of add decay
            loss_dict['l2'] = l2
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l2'] + loss_dict['kl'] * self.kl_weight
            return all_l2, loss_dict
        else: # inference time
            future_features = future_features[:, :self.model.num_queries]
            if len(future_features.shape) == 4:
                future_features = future_features.squeeze(2)
            is_pad = is_pad[:, :self.model.num_queries]
            feature_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            loss_dict = dict()
            # future_features = future_features.squeeze(-2)
            print(future_features.shape, " future_features\n")
            print(feature_hat.shape, " feature_hat\n")
            all_l2 = F.mse_loss(future_features, feature_hat, reduction='none')
            l2 = (all_l2 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l2'] = l2
            loss_dict['loss'] = loss_dict['l2']
            return all_l2, loss_dict

    def configure_optimizers(self):
        return self.optimizer

class MINEEstimator(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_MINE_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

    def __call__(self, image, action, time_step, qpos=None):
        loss_dict = dict()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        # old_image = image
        image = normalize(image)
        positive_mine_outputs = self.model(image, action, time_step, qpos=qpos)
        negative_mine_outputs = self.model(image, action, time_step, qpos=qpos, random_timestep=True)
        # positive_magnitude_loss = self.magnitude_reg_loss(positive_mine_outputs)
        # negative_magnitude_loss = self.magnitude_reg_loss(negative_mine_outputs)
        positive_loss = torch.mean(positive_mine_outputs)
        negative_loss = torch.logsumexp(negative_mine_outputs, dim=0) - np.log(negative_mine_outputs.shape[0])
        # anchor_loss = F.relu(negative_mine_outputs.mean())
        anchor_loss = torch.mean(negative_mine_outputs ** 2)
        mine_loss = negative_loss - positive_loss
        total_loss = mine_loss + 1 * anchor_loss
        loss_dict['loss'] = mine_loss
        loss_dict['anchor_loss'] = anchor_loss
        loss_dict['total_loss'] = total_loss
        
        print(torch.mean(torch.exp(positive_mine_outputs)).item(), "-------EXP_POSITIVE_MINE-------")
        print(torch.mean(torch.exp(negative_mine_outputs)).item(), "-------EXP_NEGATIVE_MINE-------")
        return loss_dict, torch.exp(positive_mine_outputs), torch.exp(negative_mine_outputs)
    
    def magnitude_reg_loss(self, out_batch, log_low=-2, log_high=2):
        magnitude = out_batch / np.log(10)
        hinge = torch.nn.functional.relu(log_low - magnitude) + torch.nn.functional.relu(magnitude - log_high)
        return hinge.mean()
        
    
    def forward_sequence(self, image, action_seq, qpos, is_pad=None):
        # if is_pad is not None:
        #     # is_pad = is_pad[:, :action_seq.shape[1]]
        #     pad_start = 100 - torch.sum(is_pad[0]).item()
        #     pad_start = str(pad_start) if pad_start < 100 else 'NA'
        #     ts = torch.randint(0, action_seq.shape[1], (1,))
        #     test_action = action_seq[:, ts.item()]
        #     test_output = self.model(image, test_action, ts, qpos=qpos, test_mode=True)
        #     true_timestep = ts[0].item()
        #     import matplotlib.pyplot as plt
        #     weights = torch.exp(test_output)
        #     plot_weight = weights.cpu().detach().numpy()[0]
        #     fig, ax = plt.subplots(1,)
        #     ax.plot(plot_weight)
        #     ax.scatter(true_timestep, plot_weight[true_timestep], color='red')
        #     ax.set_xlabel("Timestep")
        #     ax.set_ylabel("Weight")
        #     ax.set_title("MINE weights")
        #     fig.savefig(f"mine_weights_ts={true_timestep}_padstart={pad_start}.png")
        #     plt.cla()
        mine_outputs = self.model.forward_sequence(image, action_seq, qpos)
        if len(mine_outputs.shape) == 3:
            probs = torch.exp(mine_outputs)
            normalized_weights = probs / torch.sum(probs, dim=2, keepdim=True) * self.model.max_timestep
            output = torch.diagonal(normalized_weights, dim1=1, dim2=2)
            return output
        else:
            return torch.exp(mine_outputs) # [batch_size, num_queries]

    def configure_optimizers(self):
        return self.optimizer

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.decay_rate = args_override['decay_rate']
        self.reweight = args_override['reweight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, predict_model=None,image_features=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        old_image = image
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            # all_l1_with_decay
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            ## add decay
            decay_factor = self.decay_rate ** torch.arange(all_l1.shape[-2], device=all_l1.device)

            if predict_model is not None:
                with torch.inference_mode():
                    # predict_model(qpos_data, image_data, image_rep, is_pad)
                    # Compute weights w_i for each timestep i:
                    # w_i = exp(-prediction_loss_i/(mean_prediction_loss) * 5)) * N / sum_j(exp(-prediction_loss_j/((mean_prediction_loss) * 5))) * mask_i
                    # where prediction_loss_i is the prediction loss at timestep i
                    # mean_i(prediction_loss) is the mean loss across timesteps for batch i
                    # N is the sequence length
                    # mask_i is the padding mask (1 for real data, 0 for padding)

                    loss, L = predict_model(qpos, old_image,image_features,is_pad)
                    loss = loss.sum(-1)  # Sum across feature dimensions

                    # Compute normalized importance weights
                    # weight = torch.exp(-loss / torch.mean(loss, dim=-1, keepdim=True) * 5)
                    # weight = weight / weight.sum(dim=-1, keepdim=True) * loss.shape[-1]
                    # weight = weight * (~is_pad)  # Zero out weights for padded timesteps
                    weight = torch.exp(torch.exp(-loss * 0.01))
                    weight = weight * (~is_pad)  # Zero out weights for padded timesteps

                # Clone weight to create a normal tensor that can be used in autograd
                weight = weight.clone().detach()

                print(all_l1.shape, " all_l1\n")
                print(weight, " weight\n")
                # print(decay_factor.shape, " decay_factor\n")
                # weight = weight.unsqueeze(-1)
                if self.reweight:
                    all_l1 = torch.einsum('bnd,bn->bnd', all_l1, weight)

                # all_l1 = all_l1 * weight


                    # print(L['l2'], " l2\n")
                    # print(loss.shape, " loss\n")
                    # print(weight.detach().cpu().numpy(), end=" weight\n")
                    # print(loss,end=" loss\n")
                    # exit(0)


            else:
                # print(all_l1.shape, " all_l1\n")
                # print(decay_factor.shape, " decay_factor\n")
                all_l1 = torch.einsum('bnd,n->bnd', all_l1, decay_factor)
                # all_l1= all_l1 * decay_factor


            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            ### end of add decay
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer
    
class ACTPolicy2(nn.Module):
    def __init__(self, args_override, pred_model):
        super().__init__()
        model, optimizer = build_ACT2_model_and_optimizer(args_override, pred_model)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.pred_weight = args_override['pred_weight']
        self.decay_rate = args_override['decay_rate']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, image_features=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        old_image = image
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            # all_l1_with_decay
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            ## add decay
            decay_factor = self.decay_rate ** torch.arange(all_l1.shape[-1], device=all_l1.device)
            
            # with torch.inference_mode(): # retain grad for prediction model optimizaiton
            loss, L = self.model.pred_model(qpos, old_image, image_features, is_pad)
            loss = loss.sum(-1)  # Sum across feature dimensions

            weight = torch.exp(torch.exp(-loss * 0.01))
            weight = weight * (~is_pad)  # Zero out weights for padded timesteps

            # Clone weight to create a normal tensor that can be used in autograd
            weight = weight.clone().detach()

            print(all_l1.shape, " all_l1\n")
            print(weight.shape, " weight\n")
            # print(decay_factor.shape, " decay_factor\n")
            # weight = weight.unsqueeze(-1)
            # TODO: reweighting
            # all_l1 = torch.einsum('bnd,bn->bnd', all_l1, weight)

            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            ### end of add decay
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['pred_loss'] = (loss * ~is_pad).mean()
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight + loss_dict['pred_loss'] * self.pred_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class ACTPolicy_PMI(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.decay_rate = args_override['decay_rate']
        self.reweight = args_override['reweight']
        self.weight_clip = args_override['weight_clip']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None, mine_model=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        old_image = image
        image = normalize(image)
        bs = qpos.shape[0]
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            # all_l1_with_decay
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            ## add decay
            decay_factor = self.decay_rate ** torch.arange(all_l1.shape[-2], device=all_l1.device)

            if mine_model is not None:
                with torch.inference_mode():
                    action_len = actions.shape[1]
                    action_dim = actions.shape[2]
                    c, h, w = image.shape[2:]
                    # extended_image = image.repeat_interleave(action_len, dim=0) # bs, len, c, h, w
                    exp_pmis = mine_model.forward_sequence(image, actions, qpos, is_pad)
                    # print("---WEIGHT---", (exp_pmis * (~is_pad)).clone().detach())
                    if self.weight_clip:
                        print("Clipping weights")
                        # exp_pmis = torch.clip(exp_pmis, min=0.8, max=1.2)
                        exp_pmis = torch.clip(exp_pmis, min=0.0, max=2.0)
                    # exp_pmis = exp_pmis.view(bs, action_len)
                    weight = exp_pmis * (~is_pad)  # Zero out weights for padded timesteps

                # Clone weight to create a normal tensor that can be used in autograd
                weight = weight.clone().detach()

                print(all_l1.shape, " all_l1\n")
                print(weight, " weight\n")
                # print(decay_factor.shape, " decay_factor\n")
                # weight = weight.unsqueeze(-1)
                if self.reweight:
                    all_l1 = torch.einsum('bnd,bn->bnd', all_l1, weight)

                # all_l1 = all_l1 * weight


                    # print(L['l2'], " l2\n")
                    # print(loss.shape, " loss\n")
                    # print(weight.detach().cpu().numpy(), end=" weight\n")
                    # print(loss,end=" loss\n")
                    # exit(0)


            else:
                # print(all_l1.shape, " all_l1\n")
                # print(decay_factor.shape, " decay_factor\n")
                all_l1 = torch.einsum('bnd,n->bnd', all_l1, decay_factor)
                # all_l1= all_l1 * decay_factor


            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            ### end of add decay
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]

            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
