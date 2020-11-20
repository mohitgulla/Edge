import os
import glob
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from collections import defaultdict
from utils.post_training_quantization.rounding import *
from utils.post_training_quantization.stochastic_rounding import *

'''
Quantization Aware Training
'''

'''
Given all weights for a model, the precision and approach,
the function calculates a tensor of unique values per layer (weight+bias)
Returns unique values for the entire model.
'''


def qat_unique_value_generator(weights, precision, bin_method):
    # Parse layers names
    network_name = list(set([x.rsplit('.', 1)[0] for x in list(weights.keys())]))

    unique_values = dict()

    for i in network_name:
        # uniform range based on min and max
        if bin_method == 'uniform_range':
            min_val = torch.min(weights[i+'.weight']).item()
            max_val = torch.max(weights[i+'.weight']).item()
            unique_intm = torch.linspace(start=min_val, end=max_val, steps=2**precision)
            unique_values[i+'.weight'] = unique_intm

        # Histogram based bins. Midpoint of bin edges is used as unique values
        if bin_method == 'histogram':
            bin_edges = np.histogram_bin_edges(
                weights[i+'.weight'].detach().cpu().numpy(), bins=2**precision)
            bin_edges = list(map(lambda i, j: (i + j)/2, bin_edges[: -1], bin_edges[1:]))
            unique_intm = torch.tensor(np.array(bin_edges))
            unique_values[i+'.weight'] = unique_intm

        # Range based on quantiles of a normal distribution
        if bin_method == 'prior_normal':
            mean_val = torch.mean(weights[i+'.weight']).item()
            std_val = torch.std(weights[i+'.weight']).item()
            quantiles = np.linspace(0, 1, num=2**precision)  # Quantile
            quantiles = quantiles[:-1][1:]  # Removing 1st and last element
            unique_intm = list(map(lambda x: norm.ppf(x, loc=mean_val, scale=std_val), quantiles))
            unique_intm.append(torch.min(weights[i+'.weight']).item())
            unique_intm.append(torch.max(weights[i+'.weight']).item())
            unique_intm = torch.tensor(np.array(unique_intm))
            unique_values[i+'.weight'] = unique_intm

    return unique_values


'''
Primary QAT Function which needs to be called by the user
-weights_fp is the input floating point weights of the model. Hence layer names would be present
-Precision value is an int.
-quant_method is a string. It can be either of the following : 'stochastic_rounding', 'normal_rounding', 'mid-rise'
-No bin_method is required for quant_method='mid_rise'
-bin_method is a string. It can be either of the following : 'uniform_range', 'histogram', 'prior_normal'
-combinations=[['stochastic_rounding', 'uniform_range'],
              ['stochastic_rounding', 'prior_normal'],
              ['stochastic_rounding', 'histogram'],
              ['normal_rounding', 'uniform_range'],
              ['normal_rounding', 'prior_normal'],
              ['normal_rounding', 'histogram'],
              ['mid_rise', None]]
Primary Quantization Function which needs to be called by the user
'''


def qat_quantization(weights_fp, precision, quant_method, bin_method=None):

    # Stochastic Rounding
    if (quant_method == 'stochastic_rounding') and (bin_method != None):
        weights_q = dict()
        w_fp_min_max = defaultdict(dict)
        w_q_min_max = defaultdict(dict)

        # Generates unique values per layer. Returns unique values of all layers
        unique_values = qat_unique_value_generator(weights_fp, precision, bin_method)

        for w in weights_fp:
            if w.rsplit('.', 1)[1] == 'weight':
                w_fp_min_max[w]['min'] = torch.min(weights_fp[w]).item()
                w_fp_min_max[w]['max'] = torch.max(weights_fp[w]).item()
                weights_q[w] = stochastic_quant(weights_fp[w], unique_values[w])
                w_q_min_max[w]['min'] = torch.min(weights_q[w]).item()
                w_q_min_max[w]['max'] = torch.max(weights_q[w]).item()

            else:
                weights_q[w] = weights_fp[w]  # Bias remains unquantized

    # Normal Rounding
    if (quant_method == 'normal_rounding') and (bin_method != None):
        weights_q = dict()
        w_fp_min_max = defaultdict(dict)
        w_q_min_max = defaultdict(dict)

        # Generates unique values per layer. Returns unique values of all layers
        unique_values = qat_unique_value_generator(weights_fp, precision, bin_method)

        for w in weights_fp:
            if w.rsplit('.', 1)[1] == 'weight':
                w_fp_min_max[w]['min'] = torch.min(weights_fp[w]).item()
                w_fp_min_max[w]['max'] = torch.max(weights_fp[w]).item()
                weights_q[w] = rounding_quant(weights_fp[w], unique_values[w])
                w_q_min_max[w]['min'] = torch.min(weights_q[w]).item()
                w_q_min_max[w]['max'] = torch.max(weights_q[w]).item()

            else:
                weights_q[w] = weights_fp[w]  # Bias remains unquantized

    # mid-rise quantization
    if (quant_method == 'mid_rise') and (bin_method == None):
        weights_q = dict()
        # Min and Max values of floating point. Weight and Bias is considered as a single layer for this calculation
        w_fp_min_max = defaultdict(dict)
        w_q_min_max = defaultdict(dict)  # Min and Max values of quantization

        for w in weights_fp:
            if w.rsplit('.', 1)[1] == 'weight':
                w_fp_min_max[w]['min'] = torch.min(weights_fp[w]).item()
                w_fp_min_max[w]['max'] = torch.max(weights_fp[w]).item()

                if len(weights_fp[w]) == 1:
                    delta = weights_fp[w] / 2**precision
                else:
                    delta = (torch.max(weights_fp[w]) - torch.min(weights_fp[w])) / 2**precision

                weights_q[w] = delta * (torch.floor(weights_fp[w]/delta) + 0.5)
                w_q_min_max[w]['min'] = torch.min(weights_q[w]).item()
                w_q_min_max[w]['max'] = torch.max(weights_q[w]).item()

            else:
                weights_q[w] = weights_fp[w]  # Bias remains unquantized

    return weights_q, w_fp_min_max, w_q_min_max


'''
Functionality Testing
weights=dict()
weights['net.0.weight']=torch.randn(10,10)
weights['net.0.bias']=torch.randn(10)
weights['net.1.weight']=torch.randn(5,5)
weights['net.1.bias']=torch.randn(5)
combinations=[['stochastic_rounding', 'uniform_range'],
              ['stochastic_rounding', 'prior_normal'],
              ['stochastic_rounding', 'histogram'],
              ['normal_rounding', 'uniform_range'],
              ['normal_rounding', 'prior_normal'],
              ['normal_rounding', 'histogram'],
              ['mid_rise', None]]
for i in combinations:
  print(" Quant Technique : ", i[0] , ", Mapping Technique : ", i[1])
  x,y,z=qat_quantization (weights, 2, i[0], i[1])

  for i in x:
    intm = x[i].detach().cpu()
    print(i, " : ", len(np.unique(intm)))
Results:
Quant Technique :  stochastic_rounding , Mapping Technique :  uniform_range
net.0.weight  :  4
net.0.bias  :  10
net.1.weight  :  4
net.1.bias  :  5
 Quant Technique :  stochastic_rounding , Mapping Technique :  prior_normal
net.0.weight  :  4
net.0.bias  :  10
net.1.weight  :  4
net.1.bias  :  5
 Quant Technique :  stochastic_rounding , Mapping Technique :  histogram
net.0.weight  :  4
net.0.bias  :  10
net.1.weight  :  4
net.1.bias  :  5
 Quant Technique :  normal_rounding , Mapping Technique :  uniform_range
net.0.weight  :  4
net.0.bias  :  10
net.1.weight  :  4
net.1.bias  :  5
 Quant Technique :  normal_rounding , Mapping Technique :  prior_normal
net.0.weight  :  4
net.0.bias  :  10
net.1.weight  :  4
net.1.bias  :  5
 Quant Technique :  normal_rounding , Mapping Technique :  histogram
net.0.weight  :  4
net.0.bias  :  10
net.1.weight  :  4
net.1.bias  :  5
 Quant Technique :  mid_rise , Mapping Technique :  None
net.0.weight  :  5
net.0.bias  :  10
net.1.weight  :  5
net.1.bias  :  5
'''

''' Helper functions for parameter copy and gradient modifications during QAT'''


def get_slope(w_fp_min_max, w_q_min_max, name):
    """ calculate the constant slope gradient """
    val = (w_q_min_max[name]['max'] - w_q_min_max[name]['min']) / (
                w_fp_min_max[name]['max'] - w_fp_min_max[name]['min'])
    return val


def clone_model_weights(model):
    """ copy the model weights to a weight matrix """
    weights_fp = dict()
    for name, params in model.named_parameters():
        weights_fp[name] = params.clone()
    return weights_fp


def quantize_model(model, weights_fp, qat_params):
    """ quantize the weights of the model using the method defined by qat_params"""
    weights_q, w_fp_min_max, w_q_min_max = qat_quantization(weights_fp, qat_params['precision'],
                                                            qat_params['quant_method'], qat_params['bin_method'])

    for name, params in model.named_parameters():
        params.data.copy_(weights_q[name])

    return w_fp_min_max, w_q_min_max


def copy_weights_to_model(model, weights_fp):
    """ copy the weight matrix to the model """
    for name, params in model.named_parameters():
        params.data.copy_(weights_fp[name])


def update_gradients(model, weights_fp, w_fp_min_max, w_q_min_max):
    """ update gradients of weights using straight through estimator """
    for name, params in model.named_parameters():
        params.grad *= get_slope(w_fp_min_max, w_q_min_max, name)
    copy_weights_to_model(model, weights_fp)


def run_qat_simulations():
    """ simulate QAT training for different values of precision"""
    quant_methods = ['mid_rise', 'normal_rounding', 'stochastic_rounding']
    bin_map = {
        'mid_rise': None,
        'normal_rounding': ['histogram', 'prior_normal', 'uniform_range'],
        'stochastic_rounding': ['histogram', 'prior_normal', 'uniform_range']
    }
    # precison_values = [2, 4, 6, 8, 10, 12, 14, 16]

    precison_values = [2, 4, 6]

    res_mp = {}
    for p in precison_values:
        for quant_method in quant_methods:
            bin_methods = bin_map[quant_method]
            ## for mid_rise type
            if bin_methods == None:
                qat_params = {
                    'quant_method': quant_method,
                    'bin_method': bin_methods,
                    'precision': p
                }
                test_acc = trigger(qat_params=qat_params)
                mp_str = quant_method + '-' + 'full_range'
                if not mp_str in res_mp:
                    res_mp[mp_str] = {"accuracy": [], "precision": []}
                res_mp[mp_str]["accuracy"].append(test_acc)
                res_mp[mp_str]["precision"].append(p)

                print(res_mp)
            else:
                for bin_method in bin_methods:
                    qat_params = {
                        'quant_method': quant_method,
                        'bin_method': bin_method,
                        'precision': p
                    }

                    test_acc = trigger(qat_params=qat_params)
                    mp_str = quant_method + '-' + bin_method

                    if not mp_str in res_mp:
                        res_mp[mp_str] = {"accuracy": [], "precision": []}
                    res_mp[mp_str]["accuracy"].append(test_acc)
                    res_mp[mp_str]["precision"].append(p)
                print(res_mp)

    return res_mp