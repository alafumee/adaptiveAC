import torch.nn as nn
from torch.nn import functional as F
import torch
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer, build_ACT_prediction_model_and_optimizer
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

    def __call__(self, qpos, image, features=None, is_pad=None):
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


class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.decay_rate = args_override['decay_rate']
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
                    loss, L = predict_model(qpos, old_image,image_features,is_pad)
                    loss = loss.sum(-1)
                    # loss = (~is_pad) * loss

                    weight = torch.exp(-loss / torch.mean(loss, dim=-1, keepdim=True) * 5)
                    weight = weight / weight.sum(dim=-1, keepdim=True) * loss.shape[-1]
                    weight = weight * (~is_pad)

                # Clone weight to create a normal tensor that can be used in autograd
                weight = weight.clone().detach()

                print(all_l1.shape, " all_l1\n")
                print(weight.shape, " weight\n")
                # print(decay_factor.shape, " decay_factor\n")
                # weight = weight.unsqueeze(-1)
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
