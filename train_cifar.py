import time
import torch
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model.dnn import DenseNeuralNet
from model.cnn import CNN
from data.churn_data import ChurnDataset
from data.telescope_data import TelescopeDataset
from utils.util_functions import *
from data.mv_data import MVDataset
from tqdm.auto import trange, tqdm
from tqdm import trange
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split



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


def evaluate(model, test_set, batch_size, criterion, ep = 0):
  test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = batch_size, shuffle=True)
  test_iterator = tqdm(test_loader, desc = 'Eval Iteration for epoch:'+str(ep+1), ncols = 900)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
   

  model.eval()
  global_step = 0
  total_correct = 0
  total_samples = 0
  total_loss = 0.0
  for step, inputs in enumerate(test_iterator):
      global_step +=1
      # if global_step > 500:
      #   break
      x = inputs[0]
      y = inputs[1].long()
      x = x.to(device)
      y = y.to(device)

      logits = model(x)
      loss = criterion(logits, y)
      correct, samples = get_accuracy(logits, y)
      total_correct +=correct.item()
      total_samples +=samples
      total_loss +=loss
  # print(total_correct, total_samples)
  acc = total_correct / total_samples
  total_loss = total_loss / global_step
  # model.train()
  
  return (total_loss, acc)


def train(model, train_set, val_set, test_set , batch_size = 16, learning_rate = 0.03, epochs = 5, eval_steps = 10, skip_train_set = True):
  criterion = nn.CrossEntropyLoss()

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  train_log = open("log/train.log", "w")
  val_log = open("log/val.log", "w")
  test_log = open("log/test.log", "w")
  
  model = model.to(device)

  # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  train_loader = torch.utils.data.DataLoader(dataset= train_set, batch_size=batch_size, shuffle=True)
  global_step = 0
  for ep in tqdm(range(epochs), desc = ' Epoch Progress:', ncols=900):
    train_iterator = tqdm(train_loader, desc = 'Train Iteration for epoch:'+ str(ep+1), ncols=900)    
    for step, inputs in enumerate(train_iterator):
      model.train()
      optimizer.zero_grad()

      global_step +=1
      # if global_step > 10:
      #   break
      x = inputs[0]
      y = inputs[1].long()
      x = x.to(device)
      y = y.to(device)
      logits = model(x)

      loss = criterion(logits, y)

      loss.backward()
      optimizer.step()

      
    val_loss, val_accuracy = evaluate(model, val_set, batch_size, criterion, ep)
    val_log.write("Epoch = {}, validation loss =  {}, validation accuracy = {} \n".format(ep+1, val_loss, val_accuracy))
    
    if not skip_train_set:
      train_loss , train_accuracy = evaluate(model, train_set, batch_size, criterion, ep)
      train_log.write("Epoch = {}, training loss =  {}, training accuracy = {} \n".format(ep+1, train_loss, train_accuracy))
      print("Step = %d, training loss =  %f, training accuracy = %f" %(global_step, train_loss, train_accuracy))

    print("Step = %d, validation loss =  %f, validation accuracy = %f" %(global_step, val_loss, val_accuracy))

  if test_set is not None:  
    test_loss, test_accuracy = evaluate(model, test_set, batch_size, criterion, ep)
    test_log.write("End of training, test loss =  {}, test accuracy = {} \n".format(test_loss, test_accuracy))
    print("End of Training, test loss =  %f, test accuracy = %f" %(test_loss, test_accuracy))

  train_log.close()
  val_log.close()
  test_log.close()

def main(load_model = False):
### config params
  input_dim =  10
  output_classes = 1
  learning_rate = 0.001
  batch_size = 4
  epochs = 5
  eval_steps = 100
  model_dir = 'model_artifacts'
  model_name = 'cifar_model.pt'
####


  train_set, val_set, test_set = None, None, None
  train_set = get_cifar_dataset(train = True)
  val_set = get_cifar_dataset(train = False)

  if load_model == True:
    model = torch.load(os.path.join(model_dir, model_name))
    criterion = nn.CrossEntropyLoss()
    val_loss, val_accuracy = evaluate(model, val_set, batch_size, criterion)
    print("Running evaluation on loaded model, validation loss = %f, validation accuracy = %f"%(val_loss, val_accuracy))

  else:
    model = CNN()
    train(model, train_set, val_set, test_set , batch_size = batch_size, learning_rate = learning_rate, epochs = epochs, eval_steps = eval_steps, skip_train_set = True)
    torch.save(model, os.path.join(model_dir, model_name))


if __name__ == "__main__":
  main(load_model = False)

