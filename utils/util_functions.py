import torch
import torchvision
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


