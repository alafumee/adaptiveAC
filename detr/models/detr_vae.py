# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer , Transformer

import numpy as np

import IPython
e = IPython.embed


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    """
    Sinusoid position encoding table
    
    Input: n_position, d_hid
    
    Output: torch.FloatTensor, shape=(1, n_position, d_hid)
    """
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, action_dim, num_queries, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            print(self.backbones, backbones,end=" backbones\n")
            # for param in self.backbones[0].parameters():
            #     assert param.requires_grad == False
                # print(param.requires_grad,end=" param\n")
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        # print(qpos.shape, " qpos\n")
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
        # print(latent_input.shape, "  latent_input\n")
        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                # print(features.shape, "  features\n")
                # print(all_cam_features[-1].shape, "  pos\n")
                all_cam_pos.append(pos)
            # proprioception features
            # print(self.camera_names,end=" camera_names\n")
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            # print(src.shape, "  src\n")
            # print(pos.shape, "  pos\n")
            # print(latent_input.shape, "  latent_input\n")
            # print(proprio_input.shape, "  proprio_input\n")
            # print(self.additional_pos_embed.weight.shape, "  additional_pos_embed\n")
            # print(self.query_embed.weight.shape, "  ")
            # print(self.transformer, "  transformer\n")
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2

            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        # print(hs.shape, "  hs\n")
        # print(a_hat.shape, "    a_hat\n")
        return a_hat, is_pad_hat, [mu, logvar]
    
    
class DETRVAE_with_model(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, pred_model, transformer, encoder, state_dim, action_dim, num_queries, camera_names):
        """ Initializes the model.
        Parameters:
            pred_model: a feature prediction model
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # if backbones is not None:
        #     self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        #     self.backbones = nn.ModuleList(backbones)
        #     self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        #     # print(self.backbones, backbones,end=" backbones\n")
        #     # for param in self.backbones[0].parameters():
        #     #     assert param.requires_grad == False
        #         # print(param.requires_grad,end=" param\n")
        # else:
        # input_dim = 14 + 7 # robot_state + env_state
        self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        self.input_proj_env_state = nn.Linear(7, hidden_dim)
        self.pos = torch.nn.Embedding(2, hidden_dim)
        self.pred_model = pred_model

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        print(qpos.shape, " qpos\n")
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
        print(latent_input.shape, "  latent_input\n")
        if self.pred_model is not None:
            src = self.pred_model.get_features(qpos, image)
            pos = get_sinusoid_encoding_table(src.shape[1], src.shape[2]).squeeze(0).to(src.device)
            # Image observation features and position embeddings
            # all_cam_features = []
            # all_cam_pos = []
            # for cam_id, cam_name in enumerate(self.camera_names):
            #     features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
            #     features = features[0] # take the last layer feature
            #     pos = pos[0]
            #     all_cam_features.append(self.input_proj(features))
            #     print(features.shape, "  features\n")
            #     print(all_cam_features[-1].shape, "  pos\n")
            #     all_cam_pos.append(pos)
            # # proprioception features
            # print(self.camera_names,end=" camera_names\n")
            proprio_input = self.input_proj_robot_state(qpos)
            # # fold camera dimension into width dimension
            # src = torch.cat(all_cam_features, axis=3)
            # pos = torch.cat(all_cam_pos, axis=3)
            # print(src.shape, "  src\n")
            # print(pos.shape, "  pos\n")
            # print(latent_input.shape, "  latent_input\n")
            # print(proprio_input.shape, "  proprio_input\n")
            # print(self.additional_pos_embed.weight.shape, "  additional_pos_embed\n")
            # print(self.query_embed.weight.shape, "  query_embed\n")
            # print(self.transformer, "  transformer\n")
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
            
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2

            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        print(hs.shape, "  hs\n")
        print(a_hat.shape, "    a_hat\n")
        return a_hat, is_pad_hat, [mu, logvar]



class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    # state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_act2(args, pred_model):
    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE_with_model(
        pred_model,
        transformer,
        encoder,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

class DynamicLatentModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = None
        self.optimizer = None
        self.latent_dim = 512
        self.state_dim = 14
        self.chunk_size = 100
        self.transformer = Transformer(d_model=self.latent_dim,
                                       nhead=8,
                                       num_encoder_layers=4,
                                       num_decoder_layers=7,
                                       dim_feedforward=2048,
                                       dropout=0.1,
                                       activation="relu",
                                       normalize_before=False)
        self.camera_names = ['top']
        # Embedding layer for state
        self.embed = nn.Embedding(self.state_dim, self.latent_dim)
        self.query_embed = nn.Embedding(self.chunk_size, self.latent_dim)
        # Backbone
        print(args, "args\n")
        backbones = []
        backbone = build_backbone(args)
        backbones = [backbone]
        self.backbones = nn.ModuleList(backbones)
        self.input_proj = nn.Conv2d(self.backbones[0].num_channels, self.latent_dim, kernel_size=1)
        self.input_proj_robot_state = nn.Linear(self.state_dim, self.latent_dim)

        self.features_head = nn.Linear(self.latent_dim, self.backbones[0].num_channels)
        # self.is_pad_head = nn.Linear(self.latent_dim, 1)
        self.additional_pos_embed = nn.Embedding(1, self.latent_dim)

    def forward(self, qpos, image):
        ## input image, qpos
        ## output: latent of chunk of images [batch_size, chunk_size, latent_dim]
         # [batch_size, latent_dim]
        all_cam_features = []
        all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
            features = features[0] # take the last layer feature
            pos = pos[0]
            all_cam_features.append(self.input_proj(features))
            # print(features.shape, "  features\n")
            # print(all_cam_features[-1].shape, "  pos\n")
            all_cam_pos.append(pos)
            # proprioception features
        # print(self.camera_names,end=" camera_names\n")
        proprio_input = self.input_proj_robot_state(qpos)
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)
        proprio_input = self.input_proj_robot_state(qpos)
        # print(src.shape, "  src\n")
        # print(pos.shape, "  pos\n")
        # print(latent_input.shape, "  latent_input\n")
        # print(proprio_input.shape, "  proprio_input\n")
        # print(self.additional_pos_embed.weight.shape, "  additional_pos_embed\n")
        # print(self.query_embed.weight.shape, "  query_embed\n")
        # print(self.transformer, "  transformer\n")
        hs = self.transformer(src, None, self.query_embed.weight, pos, None, proprio_input, self.additional_pos_embed.weight)
        # print(hs.shape, " hs\n") # 100 512 8-> 8 100 512
        hs = hs.permute(2, 0, 1)
        features = self.features_head(hs)
        return features
    def get_features(self, image):
        all_cam_features = []
        all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
            features = features[0] # take the last layer feature
            pos = pos[0]
            all_cam_features.append(self.input_proj(features))
            all_cam_pos.append(pos)
        src = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)
        feat = torch.nn.functional.adaptive_avg_pool2d(src, (1, 1)).squeeze(-1).squeeze(-1)
        print(feat.shape, "feat")
        return feat
    
def build_prediction_model(args):
    model = DynamicLatentModel(args)
    return model

class SingleStepDynamicsModel(nn.Module):
    def __init__(self, backbones, decoder, encoder, state_dim, feature_dim, camera_names, num_queries=1):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = decoder
        self.encoder = encoder
        self.hidden_dim = hidden_dim = decoder.d_model
        self.feature_head = nn.Linear(hidden_dim, feature_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            print(self.backbones, backbones,end=" backbones\n")
            # for param in self.backbones[0].parameters():
            #     assert param.requires_grad == False
                # print(param.requires_grad,end=" param\n")
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(feature_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(3, hidden_dim) # learned position embedding for timestep, proprio and latent

    def timestep_embedding(self, timesteps, max_period=10000):
        # use time embedding to encode timestep
        half_dim = self.hidden_dim // 2
        emb = np.log(max_period) / half_dim
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
        return torch.tensor(emb, dtype=torch.float32).to(timesteps.device)
    
    def forward(self, qpos, image, env_state, timestep, future_feature=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        timestep: bs, how many timesteps ahead to predict, only predict that one step
        """
        is_training = future_feature is not None # train or val
        print(qpos.shape, " qpos\n")
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # TODO: change encoder architecture to a smaller one
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(future_feature) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only # bs, hidden_dim
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample) # bs, hidden_dim
        print(latent_input.shape, "  latent_input\n")
        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                print(features.shape, "  features\n")
                print(all_cam_features[-1].shape, "  pos\n")
                all_cam_pos.append(pos)
            # proprioception features
            print(self.camera_names,end=" camera_names\n")
            proprio_input = self.input_proj_robot_state(qpos)
            timestep_token = self.timestep_embedding(timestep) # bs, hidden_dim
            other_input = torch.cat([proprio_input[None], latent_input[None], timestep_token[None]], axis=0) # 3, bs, hidden_dim
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            print(src.shape, "  src\n")
            print(pos.shape, "  pos\n")
            print(latent_input.shape, "  latent_input\n")
            print(proprio_input.shape, "  proprio_input\n")
            print(self.additional_pos_embed.weight.shape, "  additional_pos_embed\n")
            print(self.query_embed.weight.shape, "  query_embed\n")
            print(self.transformer, "  transformer\n")
            # transformer here is a decoder-only architecture
            hs = self.transformer(src, self.query_embed.weight, other_input, additional_pos_embed=self.additional_pos_embed.weight, pos_embed=pos)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2

            hs = self.transformer(transformer_input, self.query_embed.weight, pos_embed=self.pos.weight)[0]
        a_hat = self.feature_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        print(hs.shape, "  hs\n")
        print(a_hat.shape, "    a_hat\n")
        return a_hat, is_pad_hat, [mu, logvar]
    

def build_single_step_prediction_model(args):
    # state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    encoder = build_encoder(args)
    transformer = build_transformer(args)

    model = SingleStepDynamicsModel(
        backbones,
        transformer,
        encoder,
        state_dim=args.state_dim,
        feature_dim=args.action_dim,
        camera_names=args.camera_names,
        num_queries=args.num_queries,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters in prediction model: %.2fM" % (n_parameters/1e6,))

    return model

class MINENetwork(nn.Module):
    def __init__(self, action_dim, max_timestep=100, hidden_dim=256, backbone=None, state_dim=14, self_normalize=False):
        # for now we do not use qpos
        super(MINENetwork, self).__init__()
        if backbone is not None:
            self.backbone = backbone
            self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        else:
            import torchvision.models as models
            pretrained_model = models.resnet18(weights='DEFAULT')
            self.backbone = nn.Sequential(*list(pretrained_model.children())[:-2])
            
            self.input_proj = nn.Linear(512, hidden_dim)
        # self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.qpos_proj = nn.Linear(state_dim, hidden_dim)
        # self.output_proj = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        self.output_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query_token = nn.Parameter(torch.randn(hidden_dim,))
        self.bias = nn.Parameter(torch.zeros(1))
        self.max_timestep = max_timestep
        self.self_normalize = self_normalize
    
    def timestep_embedding(self, timesteps, max_period=1000):
        # use time embedding to encode timestep
        device = timesteps.device
        timesteps = timesteps.cpu().numpy()
        half_dim = self.hidden_dim // 2
        emb = np.log(max_period) / half_dim
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
        return torch.tensor(emb, dtype=torch.float32).to(device)
    
    def get_pos_embd_2d(self, height, width):
        width_embedding = get_sinusoid_encoding_table(width, self.hidden_dim // 2).repeat(height, 1, 1)
        height_embedding = get_sinusoid_encoding_table(height, self.hidden_dim // 2).repeat(width, 1, 1).transpose(0, 1)
        return torch.cat([width_embedding, height_embedding], dim=-1)
    
    def get_mine_ouput(self, pooled_image_features, actions, timesteps, qpos=None):
        action_features = self.action_proj(actions)
        qpos_features = self.qpos_proj(qpos)
        timestep_features = self.timestep_embedding(timesteps)
        features = torch.cat([pooled_image_features, action_features, qpos_features], dim=-1)
        obs_act_feat = self.mlp(features)
        # print("---VAR---", torch.std(obs_act_feat, dim=0))
        # print("---MEAN---", torch.mean(obs_act_feat, dim=0))
        # print('-----SAMPLE_FEAT-----', obs_act_feat[0])
        # print(f'-----SAMPLE_TIME{timesteps[0]}-----', timestep_features[0])
        # mine_output = torch.einsum('bc, bc -> b', obs_act_feat, timestep_features) + self.bias
        mine_output = self.output_proj(torch.cat([obs_act_feat, timestep_features], dim=-1))
        # return mine_output.unsqueeze(-1)
        return mine_output
    
    def get_mine_total_output(self, pooled_image_features, actions, qpos=None):
        # get mine output for all timesteps
        for ts in range(self.max_timestep):
            timestep = torch.tensor([ts] * pooled_image_features.size(0)).to(pooled_image_features.device)
            if ts == 0:
                mine_output = self.get_mine_ouput(pooled_image_features, actions, timestep, qpos)
            else:
                mine_output = torch.cat([mine_output, self.get_mine_ouput(pooled_image_features, actions, timestep, qpos)], dim=-1)
        return mine_output.unsqueeze(1) # bs, 1, len
        
    def forward(self, image, action, timestep, qpos=None, random_timestep=False, test_mode=False):
        # image: bs, (1), c, h, w
        # action: bs, action_dim
        # timestep: bs, 
        if len(image.size()) == 5:
            image = image.squeeze(1)
        if test_mode:
            image = image[:1]
            action = action[:1]
            qpos = qpos[:1]
            # true_timestep = timestep[0]
            bs = image.size(0)
            image_features = self.backbone(image).permute(0, 2, 3, 1) # bs, height, width, 512
            image_features = self.input_proj(image_features)
            image_pos_embd = self.get_pos_embd_2d(image_features.size(1), image_features.size(2)).unsqueeze(0).repeat(bs, 1, 1, 1).to(image.device)
            attn_map = (image_pos_embd @ self.query_token).view(bs, -1) # bs, height*width
            attn_weights = torch.softmax(attn_map, dim=-1)
            pooled_image_features = torch.einsum('bnc, bn -> bc', image_features.view(bs, -1, self.hidden_dim), attn_weights)
            for ts in range(self.max_timestep):
                timestep = torch.tensor([ts] * image.size(0)).to(image.device)
                if ts == 0:
                    mine_output = self.get_mine_ouput(pooled_image_features, action, timestep, qpos)
                else:
                    mine_output = torch.cat([mine_output, self.get_mine_ouput(pooled_image_features, action, timestep, qpos)], dim=-1)
            
        else:
            if random_timestep:
                print(timestep)
                timestep = torch.randint(0, self.max_timestep, (image.size(0),)).to(image.device)
                print("----RESAMPLED TIMESTEP----", timestep)
            bs = image.size(0)
            image_features = self.backbone(image).permute(0, 2, 3, 1) # bs, height, width, 512
            image_features = self.input_proj(image_features)
            image_pos_embd = self.get_pos_embd_2d(image_features.size(1), image_features.size(2)).unsqueeze(0).repeat(bs, 1, 1, 1).to(image.device)
            attn_map = (image_pos_embd @ self.query_token).view(bs, -1) # bs, height*width
            attn_weights = torch.softmax(attn_map, dim=-1)
            pooled_image_features = torch.einsum('bnc, bn -> bc', image_features.view(bs, -1, self.hidden_dim), attn_weights)
            action_features = self.action_proj(action)
            qpos_features = self.qpos_proj(qpos)
            timestep_features = self.timestep_embedding(timestep)
            features = torch.cat([pooled_image_features, action_features, qpos_features], dim=-1)
            obs_act_feat = self.mlp(features)
            # diff = obs_act_feat - timestep_features
            # mine_output = self.output_proj(diff)
            # mine_output = torch.einsum('bc, bc -> b', obs_act_feat, timestep_features) + self.bias
            mine_output = self.output_proj(torch.cat([obs_act_feat, timestep_features], dim=-1))
            
        return mine_output#, obs_act_feat
    
    def forward_sequence(self, image, action_sequence, qpos=None):
        # perform MINE on a single image with each of the actions in the sequence
        # image: bs, (1), c, h, w
        # action: bs, len, action_dim
        if len(image.size()) == 5:
            image = image.squeeze(1)
        bs = image.size(0)
        image_features = self.backbone(image).permute(0, 2, 3, 1) # bs, height, width, 512
        image_features = self.input_proj(image_features)
        image_pos_embd = self.get_pos_embd_2d(image_features.size(1), image_features.size(2)).unsqueeze(0).repeat(bs, 1, 1, 1).to(image.device)
        attn_map = (image_pos_embd @ self.query_token).view(bs, -1) # bs, height*width
        attn_weights = torch.softmax(attn_map, dim=-1)
        pooled_image_features = torch.einsum('bnc, bn -> bc', image_features.view(bs, -1, self.hidden_dim), attn_weights)
        if self.self_normalize:
            for ts in range(action_sequence.size(1)):
                action = action_sequence[:, ts]
                # ts_tensor = torch.tensor([ts] * bs).to(image.device)
                # if ts == 0:
                #     mine_output = self.get_mine_ouput(pooled_image_features, action, ts_tensor, qpos)
                # else:
                #     mine_output = torch.cat([mine_output, self.get_mine_ouput(pooled_image_features, action, ts_tensor, qpos)], dim=-1)
                if ts == 0:
                    mine_output = self.get_mine_total_output(pooled_image_features, action, qpos) # bs, 1, len
                else:
                    mine_output = torch.cat([mine_output, self.get_mine_total_output(pooled_image_features, action, qpos)], dim=1)
        else:
            for ts in range(action_sequence.size(1)):
                action = action_sequence[:, ts]
                ts_tensor = torch.tensor([ts] * bs).to(image.device)
                if ts == 0:
                    mine_output = self.get_mine_ouput(pooled_image_features, action, ts_tensor, qpos)
                else:
                    mine_output = torch.cat([mine_output, self.get_mine_ouput(pooled_image_features, action, ts_tensor, qpos)], dim=-1)
        return mine_output # bs, len, len or bs, len


class MINENetwork_old(nn.Module):
    def __init__(self, action_dim, max_timestep=100, hidden_dim=256, backbone=None, state_dim=14):
        # for now we do not use qpos
        super(MINENetwork_old, self).__init__()
        if backbone is not None:
            self.backbone = backbone
            self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        else:
            import torchvision.models as models
            pretrained_model = models.resnet18(weights='DEFAULT')
            self.backbone = nn.Sequential(*list(pretrained_model.children())[:-2])
            
            self.input_proj = nn.Linear(512, hidden_dim)
        # self.backbone = backbone
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        # self.qpos_proj = nn.Linear(state_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.activation = nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            self.activation,
            nn.Linear(hidden_dim, hidden_dim),
        )
        # self.query_token = nn.Parameter(torch.randn(hidden_dim,))
        # self.bias = nn.Parameter(torch.zeros(1))
        self.max_timestep = max_timestep
    
    def timestep_embedding(self, timesteps, max_period=1000):
        # use time embedding to encode timestep
        device = timesteps.device
        timesteps = timesteps.cpu().numpy()
        half_dim = self.hidden_dim // 2
        emb = np.log(max_period) / half_dim
        emb = np.exp(np.arange(half_dim) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
        return torch.tensor(emb, dtype=torch.float32).to(device)
    
    def get_pos_embd_2d(self, height, width):
        width_embedding = get_sinusoid_encoding_table(width, self.hidden_dim // 2).repeat(height, 1, 1)
        height_embedding = get_sinusoid_encoding_table(height, self.hidden_dim // 2).repeat(width, 1, 1).transpose(0, 1)
        return torch.cat([width_embedding, height_embedding], dim=-1)
    
    def get_mine_ouput(self, pooled_image_features, actions, timesteps, qpos=None):
        action_features = self.action_proj(actions)
        # qpos_features = self.qpos_proj(qpos)
        timestep_features = self.timestep_embedding(timesteps)
        features = torch.cat([pooled_image_features, action_features], dim=-1)
        obs_act_feat = self.mlp(features)
        diff = obs_act_feat - timestep_features
        mine_output = self.output_proj(diff)
        return mine_output
    
    def forward(self, image, action, timestep, qpos=None, random_timestep=False, test_mode=False):
        # image: bs, (1), c, h, w
        # action: bs, action_dim
        # timestep: bs, 
        if len(image.size()) == 5:
            image = image.squeeze(1)
        if test_mode:
            image = image[:1]
            action = action[:1]
            true_timestep = timestep[0]
            bs = image.size(0)
            image_features = self.backbone(image).permute(0, 2, 3, 1) # bs, height, width, 512
            image_features = self.input_proj(image_features)
            image_pos_embd = self.get_pos_embd_2d(image_features.size(1), image_features.size(2)).unsqueeze(0).repeat(bs, 1, 1, 1).to(image.device)
            image_features = image_features + image_pos_embd
            pooled_image_features = torch.mean(image_features, dim=(1, 2))
            for ts in range(self.max_timestep):
                timestep = torch.tensor([ts] * image.size(0)).to(image.device)
                if ts == 0:
                    mine_output = self.get_mine_ouput(pooled_image_features, action, timestep)
                else:
                    mine_output = torch.cat([mine_output, self.get_mine_ouput(pooled_image_features, action, timestep)], dim=-1)
            weights = torch.exp(mine_output)
            print(f"TRUE{true_timestep}-----WEIGHTS for DIFFERENT TIMESTEPS-----", weights)
            # import matplotlib.pyplot as plt
            # plt.plot(weights.cpu().detach().numpy()[0])
            # plt.xlabel("Timestep")
            # plt.ylabel("Weight")
            # plt.title("MINE weights")
            # plt.savefig(f"mine_weights_{true_timestep}.png")
        else:
            if random_timestep:
                print(timestep)
                timestep = torch.randint(0, self.max_timestep, (image.size(0),)).to(image.device)
                print("----RESAMPLED TIMESTEP----", timestep)
            bs = image.size(0)
            image_features = self.backbone(image).permute(0, 2, 3, 1) # bs, height, width, 512
            image_features = self.input_proj(image_features)
            image_pos_embd = self.get_pos_embd_2d(image_features.size(1), image_features.size(2)).unsqueeze(0).repeat(bs, 1, 1, 1).to(image.device)
            image_features = image_features + image_pos_embd
            pooled_image_features = torch.mean(image_features, dim=(1, 2))
            action_features = self.action_proj(action)
            timestep_features = self.timestep_embedding(timestep)
            features = torch.cat([pooled_image_features, action_features], dim=-1)
            obs_act_feat = self.mlp(features)
            diff = obs_act_feat - timestep_features
            mine_output = self.output_proj(diff)
            # mine_output = torch.einsum('bc, bc -> b', obs_act_feat, timestep_features) + self.bias
        return mine_output
    
    def forward_sequence(self, image, action_sequence, qpos=None):
        # perform MINE on a single image with each of the actions in the sequence
        # image: bs, (1), c, h, w
        # action: bs, len, action_dim
        if len(image.size()) == 5:
            image = image.squeeze(1)
        bs = image.size(0)
        image_features = self.backbone(image).permute(0, 2, 3, 1) # bs, height, width, 512
        image_features = self.input_proj(image_features)
        image_pos_embd = self.get_pos_embd_2d(image_features.size(1), image_features.size(2)).unsqueeze(0).repeat(bs, 1, 1, 1).to(image.device)
        image_features = image_features + image_pos_embd
        pooled_image_features = torch.mean(image_features, dim=(1, 2))
        for ts in range(action_sequence.size(1)):
            action = action_sequence[:, ts]
            ts_tensor = torch.tensor([ts] * bs).to(image.device)
            if ts == 0:
                mine_output = self.get_mine_ouput(pooled_image_features, action, ts_tensor)
            else:
                mine_output = torch.cat([mine_output, self.get_mine_ouput(pooled_image_features, action, ts_tensor)], dim=-1)
        return mine_output # bs, len

def build_MINE(args):
    # state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    # backbone = build_backbone(args)
    model = MINENetwork(
        action_dim=args.action_dim,
        hidden_dim=args.hidden_dim,
        backbone=None,
        state_dim=args.state_dim,
        max_timestep=args.num_queries,
        self_normalize=args.self_normalize
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters in MINE: %.2fM" % (n_parameters/1e6,))

    return model