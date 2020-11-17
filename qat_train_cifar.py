import time
import torch
import os
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary

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


def get_cifar100_dataset(train = True):
  transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
  ])

  transform_test = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
  ])

  if train == True:
      dataset = datasets.CIFAR100(root = './data', train = train, transform=transform_test, target_transform=None, download=True)
  else:
      dataset = datasets.CIFAR100(root = './data', train = train, transform=transform_test, target_transform=None, download=True)
  
  return dataset


def get_accuracy(logits, labels):
  preds = torch.argmax(logits, axis=1)
  matches = preds == labels
  return (matches.sum(), len(labels))


def evaluate(model, test_set, batch_size, criterion, ep = 0, qat_mode = False, qat_params = None):
  test_loader = torch.utils.data.DataLoader(dataset = test_set, batch_size = batch_size, shuffle=True, num_workers=1)
  test_iterator = tqdm(test_loader, desc = 'Eval Iteration for epoch:'+str(ep+1), ncols = 900)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  
  model.eval()
  
  if qat_mode:
    weights_fp = clone_model_weights(model)
    w_fp_min_max, w_q_min_max = quantize_model(model, weights_fp, qat_params)
  
  global_step = 0
  total_correct = 0
  total_samples = 0
  total_loss = 0.0
  with torch.no_grad():
    for step, inputs in enumerate(test_iterator):
      global_step +=1
      # if global_step > 500:
      #   break
      x, y = inputs[0].to(device), inputs[1].long().to(device)

      logits = model(x)
      loss = criterion(logits, y)
      correct, samples = get_accuracy(logits, y)
      total_correct +=correct.item()
      total_samples +=samples
      total_loss +=loss
  # print(total_correct, total_samples)
  acc = total_correct / total_samples
  total_loss = total_loss / global_step
  model.train()
  
  return (total_loss, acc)


def train(model, train_set, val_set, test_set , batch_size = 16, learning_rate = 0.03, epochs = 5, eval_steps = 10, skip_train_set = True, qat_mode = False, qat_params = None):
  # logging
  train_log = open("log/cifar_cnn_train.log", "a")
  val_log = open("log/cifar_cnn_val.log", "a")
  test_log = open("log/cifar_cnn_test.log", "a")

  # GPU/CPU use
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  # print("Device: ", device)
  model = model.to(device)
  # print("Model Summary:")
  # summary(model, next(iter(train_set))[0].shape)
  
  # define loss & optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

  # iterate over epoch
  weights_fp = clone_model_weights(model)
  train_loader = torch.utils.data.DataLoader(dataset= train_set, batch_size=batch_size, shuffle=True, num_workers=1)
  global_step = 0
  for ep in tqdm(range(epochs), desc = ' Epoch Progress:', ncols=900):
    train_iterator = tqdm(train_loader, desc = 'Train Iteration for epoch:'+ str(ep+1), ncols=900)    
    running_loss = 0

    # iterate over batches
    for step, inputs in enumerate(train_iterator):
      model.train()
      global_step +=1
      optimizer.zero_grad()

      loss = train_step(model, weights_fp, qat_params, inputs, criterion, optimizer, device, qat_mode = qat_mode)

      running_loss+=loss.item()

    # find validation accuracy
    val_loss, val_accuracy = evaluate(model, val_set, batch_size, criterion, ep)
    val_log.write("Epoch = {}, validation loss =  {}, validation accuracy = {} \n".format(ep+1, val_loss, val_accuracy))
    print("Step = %d, validation loss =  %.3f, validation accuracy = %.3f" %(global_step, val_loss, val_accuracy))
    
    # find train accuracy if needed
    if not skip_train_set:
      train_loss , train_accuracy = evaluate(model, train_set, batch_size, criterion, ep)
      train_log.write("Epoch = {}, training loss =  {}, training accuracy = {} \n".format(ep+1, train_loss, train_accuracy))
      print("Step = %d, training loss =  %.3f, training accuracy = %.3f" %(global_step, train_loss, train_accuracy))

  # find test accuracy with final model
  if test_set is not None:  
    test_loss, test_accuracy = evaluate(model, test_set, batch_size, criterion, ep)
    test_log.write("End of training, test loss =  {}, test accuracy = {} \n".format(test_loss, test_accuracy))
    print("End of Training, test loss =  %.3f, test accuracy = %.3f" %(test_loss, test_accuracy))

  # close log files
  train_log.close()
  val_log.close()
  test_log.close()

def trigger(qat_params = None, qat_mode = True, load_model = False):
  ### config params 
  output_classes = 100
  learning_rate = 0.001
  batch_size = 16
  epochs = 1
  eval_steps = 100
  model_dir = 'model_artifacts'
  model_name = 'cifar_cnn_model.pt'
  criterion = nn.CrossEntropyLoss()
  ####

  train_set, val_set, test_set = None, None, None
  train_set = get_cifar100_dataset(train = True)
  val_set = get_cifar100_dataset(train = False)

  if load_model == True:
    model = torch.load(os.path.join(model_dir, model_name))
    val_loss, val_accuracy = evaluate(model, val_set, batch_size, criterion)
    print("Running evaluation on loaded model, validation loss = %f, validation accuracy = %f"%(val_loss, val_accuracy))

  else:
    model = CNN(output_classes)
    train(model, train_set, val_set, test_set , batch_size = batch_size, learning_rate = learning_rate, epochs = epochs, eval_steps = eval_steps, skip_train_set = False)
    torch.save(model, os.path.join(model_dir, model_name))

  if qat_mode:
      ### QAT CODE ###
      print('# Running QAT evaluation #')
      criterion = nn.CrossEntropyLoss()
      val_loss, val_accuracy = evaluate(model, val_set, batch_size, criterion, ep = 0, qat_mode = True, qat_params = qat_params)
      print('Accuracy from quantized model = %f'%val_accuracy)
      return val_accuracy
  else: 
    # Evaluate Post Quantization
    path_result = "data/results/"
    model_results = quantization_eval_results(model_name,train_set=train_set,test_set=val_set,batch_size=batch_size,criterion=criterion, method='all')
    # print(model_results)
    model_results.to_csv(path_result + model_name[:-3]+'_v2' +".csv")


# if __name__ == "__main__":
#   main(load_model = False)
