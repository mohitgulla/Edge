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
                plt.title('Model Loss vs Precision - ' + model_title)
                plt.ylabel('Loss')

            else:
                plt.plot(obj.precision, obj.train_acc, label=k, marker='o')
                plt.title('Model Accuracy vs Precision - ' + model_title)
                plt.ylabel('Accuracy')

        plt.xlabel('Precision')
        plt.legend()

        os.chdir('../../utils/' + plots_dir_path)
        plt.savefig(str(m[:-3])+'.png', dpi=400)
        plt.clf()




    # combined = combined[combined.precision > 1]
    # churn = combined[combined.model == 'cifar_resnet50_model.pt']
    # mid_rise = churn[churn.method == 'mid-rise']
    # rounding = churn[churn.method == 'rounding']
    # stochastic = churn[churn.method == 'stochastic']
    #
    # # Accuracy
    # plt.plot(mid_rise.precision, mid_rise.train_acc, c='tab:blue', label='Mid-Rise Train', marker='o')
    # plt.plot(mid_rise.precision, mid_rise.test_acc, c='tab:blue', ls='--', label='Mid-Rise Test')
    # plt.plot(rounding.precision, rounding.train_acc, c='tab:orange', label='Rounding Train', marker='o')
    # plt.plot(rounding.precision, rounding.test_acc, c='tab:orange', ls='--', label='Rounding Test')
    # plt.plot(stochastic.precision, stochastic.train_acc, c='tab:green', label='Stochastic Train', marker='o')
    # plt.plot(stochastic.precision, stochastic.test_acc, c='tab:green', ls='--', label='Stochastic Test')
    # plt.title('Model Accuracy post Quantization - ResNet 50 on CIFAR-100 Data')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Precision')
    # plt.legend()
    # plt.show()
    #
    #
    # # Loss
    # plt.plot(mid_rise.precision, mid_rise.train_loss, c='tab:blue', label='Mid-Rise Train', marker='o')
    # plt.plot(mid_rise.precision, mid_rise.test_loss, c='tab:blue', ls='--', label='Mid-Rise Test')
    # plt.plot(rounding.precision, rounding.train_loss, c='tab:orange', label='Rounding Train', marker='o')
    # plt.plot(rounding.precision, rounding.test_loss, c='tab:orange', ls='--', label='Rounding Test')
    # plt.plot(stochastic.precision, stochastic.train_loss, c='tab:green', label='Stochastic Train', marker='o')
    # plt.plot(stochastic.precision, stochastic.test_loss, c='tab:green', ls='--', label='Stochastic Test')
    # plt.title('Model Loss post Quantization - ResNet 50 on CIFAR-100 Data')
    # plt.ylabel('Loss')
    # plt.xlabel('Precision')
    # plt.legend()
    # plt.show()

    # print(combined.groupby(['model']).groups.keys())
    # combined = combined[combined.diff_precision !=0]
    # combined = combined[combined.method == 'stochastic']
    # plt.plot(combined[combined.model == 'churn_simple.pt'].diff_precision,
    #          combined[combined.model == 'churn_simple.pt'].diff_train_acc,
    #          label = 'Churn Simple')
    # plt.plot(combined[combined.model == 'churn_complex.pt'].diff_precision,
    #          combined[combined.model == 'churn_complex.pt'].diff_train_acc,
    #          label='Churn Complex')
    # plt.plot(combined[combined.model == 'telescope_simple.pt'].diff_precision,
    #          combined[combined.model == 'telescope_simple.pt'].diff_train_acc,
    #          label='Telescope Simple')
    # plt.plot(combined[combined.model == 'telescope_complex.pt'].diff_precision,
    #          combined[combined.model == 'telescope_complex.pt'].diff_train_acc,
    #          label='Telescope Complex')
    # plt.plot(combined[combined.model == 'fmnist_cnn_model.pt'].diff_precision,
    #          combined[combined.model == 'fmnist_cnn_model.pt'].diff_train_acc,
    #          label='FMNIST Vanilla CNN')
    # plt.plot(combined[combined.model == 'fmnist_resnet9_model.pt'].diff_precision,
    #          combined[combined.model == 'fmnist_resnet9_model.pt'].diff_train_acc,
    #          label='FMNIST ResNet 9')
    # plt.plot(combined[combined.model == 'fmnist_resnet50_model.pt'].diff_precision,
    #          combined[combined.model == 'fmnist_resnet50_model.pt'].diff_train_acc,
    #          label='FMNIST ResNet 50')
    # plt.plot(combined[combined.model == 'cifar_cnn_model.pt'].diff_precision,
    #          combined[combined.model == 'cifar_cnn_model.pt'].diff_train_acc,
    #          label='CIFAR-100 Vanilla CNN')
    # plt.plot(combined[combined.model == 'cifar_resnet9_model.pt'].diff_precision,
    #          combined[combined.model == 'cifar_resnet9_model.pt'].diff_train_acc,
    #          label='CIFAR-100 ResNet 9')
    # plt.plot(combined[combined.model == 'cifar_resnet50_model.pt'].diff_precision,
    #          combined[combined.model == 'cifar_resnet50_model.pt'].diff_train_acc,
    #          label='CIFAR-100 ResNet 50')
    # plt.title('Drop in Accuracy vs. Drop in Decimal Precision for Stochastic Rounding')
    # plt.xlabel('Delta Decimal Precision')
    # plt.ylabel('Delta Accuracy')
    # plt.legend()
    # plt.show()
    # print(combined)

    return None

# plot_accuracy_vs_precision('../data/results', '../data/plots')