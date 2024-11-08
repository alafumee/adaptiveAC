# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr_vae import build as build_vae
from .detr_vae import build_cnnmlp as build_cnnmlp
from .detr_vae import build_prediction_model as build_P
from .detr_vae import build_act2 as build_act2
def build_ACT_model(args):
    return build_vae(args)

def build_CNNMLP_model(args):
    return build_cnnmlp(args)

def build_prediction_model(args):
    return build_P(args)

def build_ACT2_model(args):
    return build_act2(args)