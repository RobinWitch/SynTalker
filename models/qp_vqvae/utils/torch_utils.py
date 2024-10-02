import gc
import torch as t

def freeze_model(model):
    model.eval()
    for params in model.parameters():
        params.requires_grad = False


def unfreeze_model(model):
    model.train()
    for params in model.parameters():
        params.requires_grad = True

def zero_grad(model):
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad = None

def empty_cache():
    gc.collect()
    t.cuda.empty_cache()

def assert_shape(x, exp_shape):
    assert x.shape == exp_shape, f"Expected {exp_shape} got {x.shape}"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_state(model):
    return sum(s.numel() for s in model.state_dict().values())

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Codebook')
    parser.add_argument('--config', default='./configs/codebook.yml')
    parser.add_argument('--gpu', type=str, default='2')
    parser.add_argument('--no_cuda', type=list, default=['2'])
    parser.add_argument('--prefix', type=str, required=False, default='knn_pred_wavvq')
    parser.add_argument('--save_path', type=str, required=False, default="./Speech2GestureMatching/output/")
    parser.add_argument('--code_path', type=str, required=False)
    parser.add_argument('--VQVAE_model_path', type=str, required=False)
    parser.add_argument('--BEAT_path', type=str, default="../dataset/orig_BEAT/speakers/")
    parser.add_argument('--save_dir', type=str, default="../dataset/BEAT")
    parser.add_argument('--step', type=str, default="1")
    parser.add_argument('--stage', type=str, default="train")
    args = parser.parse_args()
    return args

