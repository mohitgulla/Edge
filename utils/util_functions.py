import torch
import random
import numpy as np
import pandas as pd

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


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


def quantization(model_name, method='all'):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    precision = [16, 12, 8, 4, 2, 1, 0]
    model_object = 'model_artifacts/' + model_name
    results = pd.DataFrame(columns=['model', 'method', 'precision', 'model_artifact',
                                    'train_loss', 'train_acc', 'test_loss', 'test_acc'])
    model = torch.load(model_object, map_location=torch.device(device))
    results = results.append(
        {'model': model_name, 'method': 'no rounding', 'precision': 32, 'model_artifact': model},
        ignore_index=True
    )

    # normal rounding
    if method == 'rounding' or method == 'all':
        for p in precision:
            weights = dict()
            model = torch.load(model_object, map_location=torch.device(device))

            for name, params in model.named_parameters():
                weights[name] = params.clone()
            for w in weights:
                weights[w] = torch.round(weights[w] * 10**p) / (10**p)
            for name, params in model.named_parameters():
                params.data.copy_(weights[name])

            results = results.append(
                {'model': model_name, 'method': 'rounding', 'precision': p, 'model_artifact': model},
                ignore_index=True
            )

    # mid-rise quantization
    if method == 'mid-rise' or method == 'all':
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
                {'model': model_name, 'method': 'mid-rise', 'precision': p, 'model_artifact': model},
                ignore_index=True
            )

    # stochastic rounding
    if method == 'stochastic' or method == 'all':
        for p in precision:
            weights = dict()
            model = torch.load(model_object, map_location=torch.device(device))

            for name, params in model.named_parameters():
                weights[name] = params.clone()
            for w in weights:
                fix = torch.sign(weights[w]) * 10**p
                weights[w] = weights[w] * fix
                diff = weights[w] - torch.floor(weights[w])
                round = torch.floor(weights[w]) + torch.tensor(np.random.binomial(1, diff.cpu().data.numpy()),device=device)
                weights[w] = round / fix
            for name, params in model.named_parameters():
                params.data.copy_(weights[name])

            results = results.append(
                {'model': model_name, 'method': 'stochastic', 'precision': p, 'model_artifact': model},
                ignore_index=True
            )

    return results
