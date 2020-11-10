import os
import glob
import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from collections import defaultdict


def get_cifar_dataset(train = True):
  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
  ])

  transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


  if train == True:
      dataset = datasets.CIFAR10(root = './data', train = train, transform=transform, target_transform=None, download=True)
  else:
      dataset = datasets.CIFAR10(root = './data', train = train, transform=transform, target_transform=None, download=True)

  return dataset

def split_data(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets

def get_get_train_val_test(dataset, val_split=0.4):
  subset = split_data(dataset, val_split = val_split)
  eval_set = split_data(subset['val'], val_split = 0.5)

  train_set = subset['train']
  val_set = eval_set['train']
  test_set = eval_set['val']

  return (train_set, val_set, test_set)


def get_accuracy(logits, labels):
  preds = torch.argmax(logits, axis=1)
  matches = preds == labels
  return (matches.sum(), len(labels))


'''
Given all weights for a model, the precision and approach,
the function calculates a tensor of unique values per layer (weight+bias)
Returns unique values for the entire model.
'''

def unique_value_generator(weights, precision, approach):
  #Parse layers names
  network_name=list(set([x.rsplit('.',1)[0] for x in list(weights.keys())]))

  unique_values=dict()
  for i in network_name:
    weight_bias_comb=torch.cat((torch.flatten(weights[i+'.weight']),
                                weights[i+'.bias']),dim=0) #Flattening the weights tensor and concatenating it with bias

    #uniform range based on min and max
    if approach =='uniform_range':
      min_val=torch.min(weight_bias_comb).item()
      max_val=torch.max(weight_bias_comb).item()
      unique_values[i+'.weight']=torch.linspace(start = min_val, end = max_val, steps = 2**precision)
      unique_values[i+'.bias']=torch.linspace(start = min_val, end = max_val, steps = 2**precision)

    #uniform range post outlier removal
    if approach == 'uniform_range_IQR':
      weight_bias_comb=np.sort(weight_bias_comb.detach().cpu().numpy())
      Q1 = np.percentile(weight_bias_comb, 25, interpolation = 'midpoint')
      Q2 = np.percentile(weight_bias_comb, 50, interpolation = 'midpoint')
      Q3 = np.percentile(weight_bias_comb, 75, interpolation = 'midpoint')
      IQR=Q3-Q1
      low_lim = Q1 - 1.5 * IQR
      up_lim = Q3 + 1.5 * IQR
      unique_values[i+'.weight']=torch.linspace(start = low_lim, end = up_lim, steps = 2**precision)
      unique_values[i+'.bias']=torch.linspace(start = low_lim, end = up_lim, steps = 2**precision)

    #Histogram based bins. Midpoint of bin edges is used as unique values
    if approach == 'histogram':
      bin_edges=np.histogram_bin_edges(weight_bias_comb.detach().cpu().numpy(), bins=2**precision)
      bin_edges=list(map(lambda i, j : (i + j)/2, bin_edges[: -1], bin_edges[1: ]))
      unique_values[i+'.weight']=torch.tensor(np.array(bin_edges))
      unique_values[i+'.bias']=torch.tensor(np.array(bin_edges))

    #Range based on quantiles of a normal distribution
    if approach == 'prior_normal':
      mean_val=torch.mean(weight_bias_comb).item()
      std_val=torch.std(weight_bias_comb).item()
      quantiles=np.linspace(0, 1, num=2**precision) #Quantile
      quantiles=quantiles[:-1][1:] #Removing 1st and last element
      unique_intm = list(map(lambda x : norm.ppf(x,loc=mean_val,scale=std_val), quantiles))
      unique_intm.append(torch.min(weight_bias_comb).item())
      unique_intm.append(torch.max(weight_bias_comb).item())
      unique_intm=torch.tensor(np.array(unique_intm))
      unique_values[i+'.weight']=unique_intm
      unique_values[i+'.bias']=unique_intm

  return unique_values


'''
Given a "weights" tensor and a "tensor of unique values",
This function will quantize all scalar values of "weights"
to one of unique values using stochastic quantization.
Input:
  weights = torch.tensor([[1.2,3.4], [2.6, 8.9]])
  unique_values = torch.tensor([0.5, 1.5])
'''
def stochastic_quant(weights, unique_values):
  # inner helper function
  def stochastic_helper(w):

    # i = 0
    # n = len(unique_values)

    # while(i<n and unique_values[i]<w):
    #   i+=1

    # # base case
    # if i==0: return unique_values[0]
    # elif i==n: return unique_values[n-1]

    n = len(unique_values)
    unique_values_numpy = unique_values.numpy()

    # base case
    if w<=np.min(unique_values_numpy): return unique_values[0]
    elif w>=np.max(unique_values_numpy): return unique_values[n-1]

    # general case
    mask = unique_values_numpy >= w
    unique_values_copy_upper = unique_values_numpy[mask]
    mask = unique_values_numpy <= w
    unique_values_copy_lower = unique_values_numpy[mask]

    # lower, upper = unique_values[i-1], unique_values[i]

    lower = np.max(unique_values_copy_lower)
    upper = np.min(unique_values_copy_upper)
    if(lower == upper):
       return lower

    lower_p = (upper - w)/(upper - lower)

    # print(lower_p.item())
    # print(lower.item())
    # print(upper.item())
    lower_pick = np.random.binomial(n=1, p=lower_p.item())

    return lower_pick*lower + (1-lower_pick)*upper

  # soring unique values
  # unique_values = torch.sort(unique_values.flatten()).values.cpu()
  unique_values = unique_values.clone().detach().cpu()
  # apply_ only works on cpu tensor, so making a copy on cpu
  weights1 = weights.clone().detach().cpu()
  # apply stochastic quantization to all values in weights
  weights1.apply_(stochastic_helper)
  weights1 = weights1.to(weights.device)

  return weights1


'''
Given a "weights" tensor and a "tensor of unique values",
This function will quantize all scalar values of "weights"
to the nearest unique value.
Input:
  weights = torch.tensor([[1.2,3.4], [2.6, 8.9]])
  unique_values = torch.tensor([0.5, 1.5])
'''
def rounding_quant(weights, unique_values):
  # inner helper function
  def rounding_helper(w):
    # i = 0
    # n = len(unique_values)
    # while(i<n and unique_values[i]<w):
    #   i+=1
    #
    # # base case
    # if i==0: return unique_values[0]
    # elif i==n: return unique_values[n-1]

    n = len(unique_values)
    unique_values_numpy = unique_values.numpy()
    # # base case
    if w<=np.min(unique_values_numpy): return unique_values[0]
    elif w>=np.max(unique_values_numpy): return unique_values[n-1]

    mask = unique_values_numpy >= w
    unique_values_copy_upper = unique_values_numpy[mask]
    mask = unique_values_numpy <= w
    unique_values_copy_lower = unique_values_numpy[mask]

    lower = np.max(unique_values_copy_lower)
    upper = np.min(unique_values_copy_upper)

    # lower, upper = unique_values[i-1], unique_values[i]

    if (w - lower) < (upper - w):
      return torch.from_numpy(np.array(lower))
    return torch.from_numpy(np.array(upper))

  # soring unique values
  # unique_values = torch.sort(unique_values.flatten()).values.cpu()
  unique_values = unique_values.clone().detach().cpu()
  # apply_ only works on cpu tensor, so making a copy on cpu
  weights1 = weights.clone().detach().cpu()
  # apply rounding quantization to all values in weights
  weights1.apply_(rounding_helper)
  weights1 = weights1.to(weights.device)

  return weights1


'''
Primary Quantization Function which needs to be called by the user
'''
def quantization (model_name, method ='all'):
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model_object = 'model_artifacts/' + model_name
  results = pd.DataFrame(columns=['model', 'quant_method', 'bin_method', 'precision', 'model_artifact',
                                'train_loss', 'train_acc', 'test_loss', 'test_acc'])

  precision = [16,12,10,8,6,4]
  unique_val_method=['uniform_range','uniform_range_IQR', 'prior_normal','histogram']

  # Stochastic Rounding
  if method == 'stochastic_rounding' or method == 'all':
    for i in unique_val_method:
      for p in precision:
        model = torch.load(model_object, map_location=torch.device(device))
        weights=dict()
        for name, params in model.named_parameters():
          weights[name] = params.clone()
        unique_values=unique_value_generator(weights,p,approach=i) #Generates unique values per layer. Returns unique values of all layers
        for w in weights:
          weights[w]=stochastic_quant(weights[w], unique_values[w])
        for name, params in model.named_parameters():
          params.data.copy_(weights[name])

        results = results.append({'model': model_name,
                                  'quant_method': 'stochastic_rounding',
                                  'bin_method':i,
                                  'precision': p,
                                  'model_artifact': model},
                                  ignore_index=True)
        print('Results appended for Stochastic Rounding :', i, '\t',p)


  # Normal Rounding
  if method == 'normal_rounding' or method == 'all':
    for i in unique_val_method:
      for p in precision:
        model = torch.load(model_object, map_location=torch.device(device))
        weights=dict()
        for name, params in model.named_parameters():
          weights[name] = params.clone()
        unique_values=unique_value_generator(weights,p,approach=i) #Generates unique values per layer. Returns unique values of all layers
        for w in weights:
          weights[w]=rounding_quant(weights[w], unique_values[w])

        for name, params in model.named_parameters():
          params.data.copy_(weights[name])

        results = results.append({'model': model_name,
                                  'quant_method': 'normal_rounding',
                                  'bin_method':i,
                                  'precision': p,
                                  'model_artifact': model},
                                  ignore_index=True)
        print('Results appended for Normal Rounding:', i, '\t',p)


  # mid-rise quantization with / without IQR
  if method == 'mid_rise' or method == 'all':
    for p in precision:
        weights = dict()
        model = torch.load(model_object, map_location=torch.device(device))

        for name, params in model.named_parameters():
            weights[name] = params.clone()
        for w in weights:
            if len(weights[w]) == 1:
                delta = weights[w] / 2**p
            else:
                delta = (torch.max(weights[w]) - torch.min(weights[w])) / 2**p
            weights[w] = delta * (torch.floor(weights[w]/delta) + 0.5)
        for name, params in model.named_parameters():
            params.data.copy_(weights[name])

        results = results.append(
            {'model': model_name,
             'quant_method': 'mid_rise',
             'bin_method': 'full_range',
             'precision': p,
             'model_artifact': model},
            ignore_index=True
        )
        print('Results appended for Mid-Rise :' ,p)

    for p in precision:
        weights = dict()
        model = torch.load(model_object, map_location=torch.device(device))

        for name, params in model.named_parameters():
            weights[name] = params.clone()
        for w in weights:
            if len(weights[w]) == 1:
                delta = weights[w] / 2**p
            else:
                weights_w=np.sort(weights[w].detach().cpu().numpy())
                Q1 = np.percentile(weights_w, 25, interpolation = 'midpoint')
                Q2 = np.percentile(weights_w, 50, interpolation = 'midpoint')
                Q3 = np.percentile(weights_w, 75, interpolation = 'midpoint')
                IQR=Q3-Q1
                low_lim = Q1 - 1.5 * IQR
                up_lim = Q3 + 1.5 * IQR
                delta = (torch.tensor(up_lim) - torch.tensor(low_lim)) / 2**p
            weights[w] = delta * (torch.floor(weights[w]/delta) + 0.5)
        for name, params in model.named_parameters():
            params.data.copy_(weights[name])

        results = results.append(
            {'model': model_name,
             'quant_method': 'mid_rise',
             'bin_method': 'range_IQR',
             'precision': p,
             'model_artifact': model},
            ignore_index=True
        )
        print('Results appended for Mid-Rise + IQR :' ,p)

  return results


################################################################################################
################################################################################################
  

'''
Quantization Aware Training 
'''

'''
Given all weights for a model, the precision and approach,
the function calculates a tensor of unique values per layer (weight+bias)
Returns unique values for the entire model.
'''

def qat_unique_value_generator(weights, precision, bin_method):
  #Parse layers names
  network_name=list(set([x.rsplit('.',1)[0] for x in list(weights.keys())]))

  unique_values = dict()

  for i in network_name:
    weight_bias_comb=torch.cat((torch.flatten(weights[i+'.weight']),weights[i+'.bias']),dim=0) #Flattening the weights tensor and concatenating it with bias
    #uniform range based on min and max
    if bin_method =='uniform_range':
      min_val=torch.min(weight_bias_comb).item()
      max_val=torch.max(weight_bias_comb).item()
      unique_intm = torch.linspace(start = min_val, end = max_val, steps = 2**precision)
      unique_values[i+'.weight']= unique_values[i+'.bias'] = unique_intm

    #Histogram based bins. Midpoint of bin edges is used as unique values
    if bin_method == 'histogram':
      bin_edges=np.histogram_bin_edges(weight_bias_comb.detach().cpu().numpy(), bins=2**precision)
      bin_edges=list(map(lambda i, j : (i + j)/2, bin_edges[: -1], bin_edges[1: ]))
      unique_intm = torch.tensor(np.array(bin_edges))
      unique_values[i+'.weight'] = unique_values[i+'.bias'] = unique_intm

    #Range based on quantiles of a normal distribution
    if bin_method == 'prior_normal':
      mean_val=torch.mean(weight_bias_comb).item()
      std_val=torch.std(weight_bias_comb).item()
      quantiles=np.linspace(0, 1, num=2**precision) #Quantile
      quantiles=quantiles[:-1][1:] #Removing 1st and last element
      unique_intm = list(map(lambda x : norm.ppf(x,loc=mean_val,scale=std_val), quantiles))
      unique_intm.append(torch.min(weight_bias_comb).item())
      unique_intm.append(torch.max(weight_bias_comb).item())
      unique_intm=torch.tensor(np.array(unique_intm))
      unique_values[i+'.weight'] = unique_values[i+'.bias'] = unique_intm

  return unique_values


'''

Primary QAT Function which needs to be called by the user
-weights_fp is the input floating point weights of the model. Hence layer names would be present
-Precision value is an int. 
-quant_method is a string. It can be either of the following : 'stochastic_rounding', 'normal_rounding', 'mid-rise'
-No bin_method is required for quant_method='mid_rise'
-bin_method is a string. It can be either of the following : 'uniform_range', 'histogram', 'prior_normal'

'''

def qat_quantization (weights_fp, precision, quant_method, bin_method=None):

  # Stochastic Rounding
  if (quant_method == 'stochastic_rounding') and (bin_method != None):
    weights_q=dict()
    w_fp_min_max = defaultdict(dict) 
    w_q_min_max = defaultdict(dict)

    unique_values = qat_unique_value_generator(weights_fp,precision,bin_method) #Generates unique values per layer. Returns unique values of all layers
    
    for w in weights_fp:
      w_fp_min_max[w]['min'] = torch.min(weights_fp[w]).item()
      w_fp_min_max[w]['max'] = torch.max(weights_fp[w]).item()
      weights_q[w] = stochastic_quant(weights_fp[w], unique_values[w])
      w_q_min_max[w]['min'] = torch.min(weights_q[w]).item()
      w_q_min_max[w]['max'] = torch.max(weights_q[w]).item()


  # Normal Rounding
  if (quant_method == 'normal_rounding') and (bin_method != None):
    weights_q=dict()
    w_fp_min_max = defaultdict(dict) 
    w_q_min_max = defaultdict(dict)

    unique_values = qat_unique_value_generator(weights_fp,precision,bin_method) #Generates unique values per layer. Returns unique values of all layers
    
    for w in weights_fp:
      w_fp_min_max[w]['min'] = torch.min(weights_fp[w]).item()
      w_fp_min_max[w]['max'] = torch.max(weights_fp[w]).item()
      weights_q[w]=rounding_quant(weights_fp[w], unique_values[w])
      w_q_min_max[w]['min'] = torch.min(weights_q[w]).item()
      w_q_min_max[w]['max'] = torch.max(weights_q[w]).item()

  # mid-rise quantization 
  if (quant_method == 'mid_rise') and (bin_method == None):
    weights_q = dict()
    w_fp_min_max = defaultdict(dict) # Min and Max values of floating point. Weight and Bias is considered as a single layer for this calculation
    w_q_min_max = defaultdict(dict)  # Min and Max values of quantization

    for w in weights_fp:
      w_fp_min_max[w]['min'] = torch.min(weights_fp[w]).item()
      w_fp_min_max[w]['max'] = torch.max(weights_fp[w]).item()

      if len(weights_fp[w]) == 1:
          delta = weights_fp[w] / 2**precision
      else:
          delta = (torch.max(weights_fp[w]) - torch.min(weights_fp[w])) / 2**precision

      weights_q[w] = delta * (torch.floor(weights_fp[w]/delta) + 0.5)
      w_q_min_max[w]['min'] = torch.min(weights_q[w]).item()
      w_q_min_max[w]['max'] = torch.max(weights_q[w]).item()

  
  return weights_q, w_fp_min_max, w_q_min_max


################################################################################################
################################################################################################


'''
Generate Accuracy vs Precision plots for various Quantization
Argument: From all .csv files within results directory
'''
def plot_accuracy_vs_precision(results_dir_path, plots_dir_path):
    os.chdir(results_dir_path)
    filename = [x for x in glob.glob('*.{}'.format('csv'))]
    merge = pd.concat([pd.read_csv(f) for f in filename])
    # merge.to_csv('combined_results.csv', index=False)
    merge = pd.read_csv('combined_results.csv') # will be commented
    merge = merge.drop(['Unnamed: 0'], axis=1)
    merge['method'] = merge.quant_method + '_' + merge.bin_method
    model_name = list(merge.groupby(['model']).groups.keys())

    for m in model_name:
        model = merge[merge.model == m]
        model_title = ' '.join(m[:-3].split('_')).title()
        method = list(model.groupby([model.method]).groups.keys())

        for k in method:
            obj = model[model.method == k]
            if obj.train_acc.isna().all():
                plt.plot(obj.precision, obj.train_loss, label=k, marker='o')
                plt.title('Model Loss vs. Precision - ' + model_title)
                plt.ylabel('Loss')

            else:
                plt.plot(obj.precision, obj.train_acc, label=k, marker='o')
                plt.title('Model Accuracy vs. Precision - ' + model_title)
                plt.ylabel('Accuracy')

        plt.xlabel('Precision')
        plt.legend()

        os.chdir('../../utils/' + plots_dir_path)
        plt.savefig(str(m[:-3])+'.png', dpi=400)
        plt.clf()

    return None

'''
Generate Drop in Accuracy vs. Drop in Precision plots for various Quantization
Argument: From combined .csv result file
'''
def plot_delta_accuracy_vs_delta_precision(combined_results_file, plots_dir_path):
    merge = pd.read_csv(combined_results_file)
    merge = merge.drop(['Unnamed: 0'], axis=1)

    # filter for classification models
    regression = ['california_simple.pt', 'california_complex.pt', 'mv_simple.pt', 'mv_complex.pt']
    merge = merge[~merge.model.isin(regression)]

    # compute delta of accuracy and precision
    merge['max_train_acc'] = merge.groupby(['model', 'method'])['train_acc'].transform('max')
    merge['diff_accuracy'] = merge.apply(lambda x: x.train_acc - x.max_train_acc, axis=1)
    merge['diff_precision'] = merge.precision.apply(lambda x: x-16)
    merge.to_csv('yo yo.csv')

    # mid-rise
    mid_rise = merge[merge.quant_method == 'mid_rise']
    method = list(mid_rise.groupby(['bin_method']).groups.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    for i in range(len(method)):
        model_name = list(mid_rise[mid_rise.bin_method == method[i]].groupby(['model']).groups.keys())
        model_title = [' '.join(name[:-3].split('_')).title() for name in model_name]

        mid_rise_method = mid_rise[mid_rise.bin_method == method[i]]
        for j in range(len(model_name)):
            axes[i].plot(mid_rise_method[mid_rise_method.model == model_name[j]].diff_precision,
                         mid_rise_method[mid_rise_method.model == model_name[j]].diff_accuracy,
                         label = model_title[j])
        axes[i].set_title('Drop in Accuracy vs. Precision for Mid-Rise ' + method[i])
        axes[i].set_xlabel('Change in Precision (wrt 16)')
        axes[i].set_ylabel('Accuracy Drop')

    axes[0].legend()
    os.chdir('../utils/' + plots_dir_path)
    plt.savefig('mid_rise_delta.png', dpi=400)
    plt.clf()

    # rounding
    rounding = merge[merge.quant_method == 'normal_rounding']
    method = list(rounding.groupby(['bin_method']).groups.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for i in range(len(method)):
        model_name = list(rounding[rounding.bin_method == method[i]].groupby(['model']).groups.keys())
        model_title = [' '.join(name[:-3].split('_')).title() for name in model_name]

        rounding_method = rounding[rounding.bin_method == method[i]]
        if i == 0:
            x = y = 0
        if i == 1:
            x = 0
            y = 1
        if i == 2:
            x = 1
            y = 0
        if i == 3:
            x = y = 1

        for j in range(len(model_name)):
            axes[x, y].plot(rounding_method[rounding_method.model == model_name[j]].diff_precision,
                         rounding_method[rounding_method.model == model_name[j]].diff_accuracy,
                         label=model_title[j])
        axes[x, y].set_title('Drop in Accuracy vs. Precision for Normal Rounding ' + method[i])
        axes[x, y].set_xlabel('Change in Precision (wrt 16)')
        axes[x, y].set_ylabel('Accuracy Drop')

    axes[1, 1].legend()
    plt.savefig('normal_rounding_delta.png', dpi=400)
    plt.clf()

    # stochastic
    stochastic = merge[merge.quant_method == 'stochastic_rounding']
    method = list(stochastic.groupby(['bin_method']).groups.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for i in range(len(method)):
        model_name = list(stochastic[stochastic.bin_method == method[i]].groupby(['model']).groups.keys())
        model_title = [' '.join(name[:-3].split('_')).title() for name in model_name]

        stochastic_method = stochastic[stochastic.bin_method == method[i]]
        if i == 0:
            x = y = 0
        if i == 1:
            x = 0
            y = 1
        if i == 2:
            x = 1
            y = 0
        if i == 3:
            x = y = 1

        for j in range(len(model_name)):
            axes[x, y].plot(stochastic_method[stochastic_method.model == model_name[j]].diff_precision,
                            stochastic_method[stochastic_method.model == model_name[j]].diff_accuracy,
                            label=model_title[j])
        axes[x, y].set_title('Drop in Accuracy vs. Precision for Stochastic Rounding ' + method[i])
        axes[x, y].set_xlabel('Change in Precision (wrt 16)')
        axes[x, y].set_ylabel('Accuracy Drop')

    axes[1, 1].legend()
    plt.savefig('stochastic_rounding_delta.png', dpi=400)
    plt.clf()

    return None
