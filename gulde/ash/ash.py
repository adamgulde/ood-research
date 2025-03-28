import os

import numpy as np
import torch
import torch.nn.functional as F


def get_msp_score(logits):
    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores


def get_energy_score(logits):
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores


def get_score(logits, method):
    if method == "msp":
        return get_msp_score(logits)
    if method == "energy":
        return get_energy_score(logits)
    exit('Unsupported scoring method')


def ash_b(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    # calculate the number of elements in each sample excluding the batch dimension
    n = x.shape[1:].numel()
    
    # calculate the number of elements to keep based on the percentile
    k = n - int(np.round(n * percentile / 100.0))
    
    # reshape the input tensor to 2D (batch_size, num_elements)
    t = x.view((b, c * h * w))
    
    # get the top k values and their indices along the last dimension
    v, i = torch.topk(t, k, dim=1)
    
    # calculate the fill value as the sum of the input per sample divided by k
    fill = s1 / k
    
    # expand the fill value to match the shape of the top k values
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    
    # zero out the tensor and scatter the fill values back into their original positions
    t.zero_().scatter_(dim=1, index=i, src=fill)
    
    return x


def ash_p(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100

    b, c, h, w = x.shape

    # calculate the number of elements in each sample excluding the batch dimension
    n = x.shape[1:].numel()
    
    # calculate the number of elements to keep based on the percentile
    k = n - int(np.round(n * percentile / 100.0))
    
    # reshape the input tensor to 2D (batch_size, num_elements)
    t = x.view((b, c * h * w))
    
    # get the top k values and their indices along the last dimension
    v, i = torch.topk(t, k, dim=1)
    
    # zero out the tensor and scatter the top k values back into their original positions
    t.zero_().scatter_(dim=1, index=i, src=v)

    return x


def ash_s(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x


def ash_rand(x, percentile=65, r1=0, r2=10):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the number of elements in each sample excluding the batch dimension
    n = x.shape[1:].numel()
    
    # calculate the number of elements to keep based on the percentile
    k = n - int(np.round(n * percentile / 100.0))
    
    # reshape the input tensor to 2D (batch_size, num_elements)
    t = x.view((b, c * h * w))
    
    # get the top k values and their indices along the last dimension
    v, i = torch.topk(t, k, dim=1)
    
    # fill the top k values with random values uniformly distributed between r1 and r2
    v = v.uniform_(r1, r2)
    
    # zero out the tensor and scatter the random values back into their original positions
    t.zero_().scatter_(dim=1, index=i, src=v)
    return x


def react(x, threshold):
    x = x.clip(max=threshold)
    return x


def react_and_ash(x, clip_threshold, pruning_percentile):
    x = x.clip(max=clip_threshold)
    x = ash_s(x, pruning_percentile)
    return x


def apply_ash(x, method):
    if method.startswith('react_and_ash@'):
        [fn, t, p] = method.split('@')
        return eval(fn)(x, float(t), int(p))

    if method.startswith('react@'):
        [fn, t] = method.split('@')
        return eval(fn)(x, float(t))

    if method.startswith('ash'):
        [fn, p] = method.split('@')
        return eval(fn)(x, int(p))

    return x
