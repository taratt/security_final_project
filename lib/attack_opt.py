import torch
import torch.nn as nn
from sympy import print_rcode

from .layerwrapper import WrappedGPT
from .data import get_loaders
import random
import numpy as np


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask


def bfa(element):
    binary_representation = element.view(torch.uint16).item()
    bit_to_flip = random.randint(0, 9)
    flipped_binary = binary_representation ^ (1 << bit_to_flip)
    flipped_value = torch.tensor(flipped_binary, dtype=torch.uint16).view(torch.float16)

    return flipped_value


def perform_bfa_on_weights(weight, percentage):
    flat_weight = weight.flatten()
    num_elements = flat_weight.numel()
    num_to_attack = int((percentage / 100) * num_elements)

    indices_to_attack = torch.randperm(num_elements)[:num_to_attack]
    for i, idx in enumerate(indices_to_attack):
        flat_weight[idx] = bfa(flat_weight[idx])

    return flat_weight.view(weight.shape)

def flip_defend(weight):

    pass

def compare_and_print_differences(tensor1, tensor2):
    """
    Compare two tensors element-wise and print the elements that are different in both tensors.

    Parameters:
        tensor1 (torch.Tensor): The first tensor.
        tensor2 (torch.Tensor): The second tensor.
    """
    # Compare the tensors element-wise
    mask = tensor1 != tensor2  # Boolean mask where True indicates a mismatch

    # Get the indices of the mismatched elements
    mismatched_indices = torch.nonzero(mask)

    print("Mismatched elements:")
    for idx in mismatched_indices:
        i, j = idx[0].item(), idx[1].item()  # Get row and column index (assuming 2D tensor)
        print(f"Index: {idx} -> Tensor 1: {tensor1[i, j]}, Tensor 2: {tensor2[i, j]}")

def perform_attack(args, model, tokenizer, device=torch.device("cuda:0"), bit_flip_percentage=1, first_third=False,
                   second_third=False, third_third=False, atten=False, atten_out=False, fc=False, defend= False):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    # print("loading calibration data")
    # dataloader, _ = get_loaders("c4", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, tokenizer=tokenizer)
    # print("dataset loading complete")
    # with torch.no_grad():
    #     inps, outs, attention_mask = prepare_calibration_input(model, dataloader, device)

    layers = model.model.decoder.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # if f"model.layers.{i}" in model.hf_device_map:
        #     dev = model.hf_device_map[f"model.layers.{i}"]
        #     inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        # wrapped_layers = {}
        # for name in subset:
        #     wrapped_layers[name] = WrappedGPT(subset[name])
        #
        # def add_batch(name):
        #     def tmp(_, inp, out):
        #         wrapped_layers[name].add_batch(inp[0].data, out.data)
        #
        #     return tmp
        #
        # handles = []
        # for name in wrapped_layers:
        #     handles.append(subset[name].register_forward_hook(add_batch(name)))
        # for j in range(args.nsamples):
        #     with torch.no_grad():
        #         outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        # for h in handles:
        #     h.remove()

        for name in subset:

            print(f"Attacking layer {i} name {name}")
           # org = subset[name].weight.data.clone()

            if atten and (name == "self_attn.k_proj" or name == "self_attn.q_proj" or name == "self_attn.v_proj"):
                subset[name].weight.data = perform_bfa_on_weights(subset[name].weight.data, bit_flip_percentage)
            if atten_out and (name == "self_attn.out_proj"):
                subset[name].weight.data = perform_bfa_on_weights(subset[name].weight.data, bit_flip_percentage)
            if fc and (name == "fc1" or name == "fc2"):
                subset[name].weight.data = perform_bfa_on_weights(subset[name].weight.data, bit_flip_percentage)
            if first_third and (i < 4):
                subset[name].weight.data = perform_bfa_on_weights(subset[name].weight.data, bit_flip_percentage)
            if second_third and (i >= 4 and i < 8):
                subset[name].weight.data = perform_bfa_on_weights(subset[name].weight.data, bit_flip_percentage)
            if third_third and (i >= 8):
                subset[name].weight.data = perform_bfa_on_weights(subset[name].weight.data, bit_flip_percentage)
            if ((not atten) and (not atten_out) and (not fc) and (not first_third) and (not second_third) and (
            not third_third)):
                subset[name].weight.data = perform_bfa_on_weights(subset[name].weight.data, bit_flip_percentage)

            if defend:
                #wanda-based defence
                # W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                #     wrapped_layers[name].scaler_row.reshape((1, -1)))
                # W_mask = (torch.zeros_like(W_metric) == 1)
                # sort_res = torch.sort(W_metric, dim=-1, stable=True)
                #
                # indices = sort_res[1][:, :-int(W_metric.shape[1] * 0.001)]
                # W_mask.scatter_(1, indices, True)
                #
                # subset[name].weight.data[W_mask] = 0

                #prune defence
                #subset[name].weight.data[subset[name].weight.data>1] = 0
                flip_defend(subset[name].weight.data)
            #compare_and_print_differences(subset[name].weight.data, org)

        # for j in range(args.nsamples):
        #     with torch.no_grad():
        #         outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        # inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
