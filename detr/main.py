# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model,build_prediction_model, build_ACT2_model, build_mine_model

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')

    # Model parameters
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--use_predict_model', action='store_true')
    parser.add_argument('--num_epochs_prediction', action='store', type=int, help='num_epochs_prediction', required=False)

    ## newly added by yuyue
    parser.add_argument('--query_freq', action='store', type=int, help='query_freq', required=False) # chunk size at eval
    parser.add_argument('--decay_rate', action='store', type=float, help='decay_rate', required=False)

    # new
    parser.add_argument('--state_dim', action='store', type=int, help='state_dim', required=False)
    parser.add_argument('--action_dim', action='store', type=int, help='action_dim', required=False)
    parser.add_argument('--prediction_ckpt_dir', action='store', type=str, help='prediction_ckpt_dir', required=False)
    parser.add_argument('--pred_weight', action='store', type=float, help='pred_weight', required=False)
    parser.add_argument('--lr_pred', action='store', type=float, help='lr_pred', required=False)
    parser.add_argument('--lr_mine', action='store', type=float, help='lr_mine', required=False)
    parser.add_argument('--load_mine', action='store_true', required=False)
    parser.add_argument('--reweight', action='store_true', required=False)
    parser.add_argument('--mine_ckpt_dir', action='store', type=str, help='mine_ckpt_dir', required=False)
    parser.add_argument('--num_epochs_mine', action='store', type=int, help='num_epochs_mine', required=False)
    parser.add_argument('--mine_batch_size', action='store', type=int, help='mine_batch_size', required=False)
    parser.add_argument('--load_mine_ckpt_path', action='store', type=str, help='load_mine_ckpt_dir', required=False)
    parser.add_argument('--self_normalize_weight', action='store_true', required=False)
    parser.add_argument('--weight_clip', action='store_false', help='weight_clip', required=False)

    return parser


def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    for k, v in args_override.items():
        setattr(args, k, v)
    # print(args.backbone, "  args.backbone\n")
    # exit(0)
    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

def build_ACT2_model_and_optimizer(args_override, pred_model):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    for k, v in args_override.items():
        setattr(args, k, v)
    # print(args.backbone, "  args.backbone\n")
    # exit(0)
    model = build_ACT2_model(args, pred_model)
    print(model)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "pred_model" not in n and "backbone" not in n and p.requires_grad]},
        {"params":[p for n, p in model.named_parameters() if "pred_model" in n and "backbone" not in n and p.requires_grad],
         "lr": args.lr_backbone},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

def build_ACT_prediction_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    for k, v in args_override.items():
        setattr(args, k, v)
    model = build_prediction_model(args)
    model.cuda()
    param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                weight_decay=args.weight_decay)

    return model, optimizer

def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer

def build_MINE_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    for k, v in args_override.items():
        setattr(args, k, v)
    
    model = build_mine_model(args)
    model.cuda()
    
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    return model, optimizer

def build_ACT_PMI_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    for k, v in args_override.items():
        setattr(args, k, v)
    model = build_ACT_model(args)
    model.cuda()
    
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    return model, optimizer

