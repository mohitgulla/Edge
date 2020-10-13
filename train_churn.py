import time
import torch
import os
from model.dnn import DenseNeuralNet
from data.churn_data import ChurnDataset
import torch.nn as nn
from utils.util_functions import *
from tqdm.auto import trange, tqdm
import numpy as np
from datetime import datetime

#Setting Random Seed
np.random.seed(0)
torch.manual_seed(0)

def evaluate(model, test_set, batch_size, criterion, ep):
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
      x = inputs[0]
      y = torch.squeeze(inputs[1],1).long()
      x = x.to(device)
      y = y.to(device)

      logits = model(x)
      loss = criterion(logits, y)
      correct, samples = get_accuracy(logits, y)
      total_correct +=correct.item()
      total_samples +=samples
      total_loss +=loss

  acc = total_correct / total_samples
  total_loss = total_loss / global_step
  
  return (total_loss, acc)

def train(model, train_set, val_set, test_set , batch_size = 16, learning_rate = 0.03, epochs = 5, eval_steps = 10, skip_train_set = True):
  criterion = nn.CrossEntropyLoss()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  time_stamp = str(datetime.now())
  filename = "Churn_Dataset_" + time_stamp
  train_log = open("log/"+ filename +"_train.log", "w")
  val_log = open("log/"+ filename +"_val.log", "w")
  test_log = open("log/"+ filename +"_test.log", "w")
  

  train_loader = torch.utils.data.DataLoader(dataset= train_set, batch_size=batch_size, shuffle=True)
  global_step = 0
  for ep in tqdm(range(epochs), desc = ' Epoch Progress:', ncols=900):
    train_iterator = tqdm(train_loader, desc = 'Train Iteration for epoch:'+ str(ep+1), ncols=900)    
    for step, inputs in enumerate(train_iterator):
      model.train()
      model.zero_grad()

      global_step +=1
      # if global_step > 10:
      #   break
      x = inputs[0]
      y = torch.squeeze(inputs[1],1).long()
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
    
  test_loss, test_accuracy = evaluate(model, test_set, batch_size, criterion, ep)
  test_log.write("End of training, test loss =  {}, test accuracy = {} \n".format(test_loss, test_accuracy))
  print("End of Training, test loss =  %f, test accuracy = %f" %(test_loss, test_accuracy))

  train_log.close()
  val_log.close()
  test_log.close()

def quantization_eval_results(model_name,train_set,test_set,batch_size,criterion):
  results = quantization(model_name)
  train_loss_list = []
  train_accuracy_list = []
  test_loss_list = []
  test_accuracy_list = []
  for i in results["model_artifact"]:
    train_loss, train_accuracy = evaluate(model=i, 
                                        test_set = train_set,
                                        batch_size=batch_size, 
                                        criterion=criterion,
                                        ep=0) 
    test_loss, test_accuracy = evaluate(model=i, 
                                        test_set = test_set,
                                        batch_size=batch_size, 
                                        criterion=criterion,
                                        ep=0)  
    train_loss_list.append(train_loss.item())
    test_loss_list.append(test_loss.item())
    train_accuracy_list.append(train_accuracy)
    test_accuracy_list.append(test_accuracy)
    # print("End of training, test loss =  {}, test accuracy = {} \n".format(train_loss, train_accuracy))
    # print("End of training, test loss =  {}, test accuracy = {} \n".format(test_loss, test_accuracy))
  results["train_loss"] = train_loss_list
  results["train_acc"] = train_accuracy_list
  results["test_loss"] = test_loss_list
  results["test_acc"] = test_accuracy_list
  return results

def main(train_model=True):
## main
  input_dim =  13
  output_classes = 2
  learning_rate = 0.001
  batch_size = 16
  epochs = 10
  eval_steps = 100
  ####
  model_dir = 'model_artifacts'
  model_simple_name = 'churn_simple.pt'
  model_complex_name = 'churn_complex.pt'
  ####
  churn_dataset = ChurnDataset()
  train_set, val_set, test_set = get_get_train_val_test(churn_dataset, 
                                                        val_split=0.40)
  if(train_model):
    print("-------------------------------------------------------")
    print("Training Model: 1")
    model_simple = DenseNeuralNet(input_size = input_dim, 
                                  num_classes = output_classes,
                                  layers = [10],
                                  dropout_prob=0,
                                  batch_norm=False)   
    print("-------------------------------------------------------")
    print(model_simple)
    print("-------------------------------------------------------")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_simple.parameters(), lr=learning_rate)

    train(model = model_simple,
          train_set = train_set, 
          val_set = val_set, 
          test_set = test_set , 
          batch_size = batch_size, 
          learning_rate = learning_rate, 
          epochs = epochs, 
          eval_steps = eval_steps,
          skip_train_set=False)  
    torch.save(model_simple, os.path.join(model_dir, model_simple_name))
    print("-------------------------------------------------------")
    print("Training Model: 2")
    model_complex = DenseNeuralNet(input_size = input_dim, 
                                  num_classes = output_classes,
                                  layers = [20,40,60,30],
                                  dropout_prob=0.10,
                                  batch_norm=False)  
    print("-------------------------------------------------------")
    print(model_complex)
    print("-------------------------------------------------------")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_complex.parameters(), lr=learning_rate)

    train(model = model_complex,
          train_set = train_set, 
          val_set = val_set, 
          test_set = test_set , 
          batch_size = batch_size, 
          learning_rate = learning_rate, 
          epochs = epochs, 
          eval_steps = eval_steps,
          skip_train_set=False)  
    torch.save(model_complex, os.path.join(model_dir, model_complex_name))
  else:
    criterion = nn.CrossEntropyLoss()
    path_result = "data/results/"

    results_simple = quantization_eval_results(model_simple_name,train_set=train_set,test_set=test_set,batch_size=batch_size,criterion=criterion)
    results_complex = quantization_eval_results(model_complex_name,train_set=train_set,test_set=test_set,batch_size=batch_size,criterion=criterion)

    results_simple.to_csv(path_result + "churn_simple.csv")
    results_complex.to_csv(path_result + "churn_complex.csv")
if __name__ == "__main__":
  main(train_model=False)