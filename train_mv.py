import time
import torch
import os
from model.dnn import DenseNeuralNet
from data.mv_data import MVDataset
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
      y = inputs[1]
      x = x.to(device)
      y = y.to(device)

      logits = model(x)
      loss = criterion(logits, y)
      total_loss +=loss

  total_loss = total_loss / global_step 
  return total_loss


def train(model, train_set, val_set, test_set , batch_size = 16, learning_rate = 0.03, epochs = 5, eval_steps = 10, skip_train_set = True):
  criterion = nn.MSELoss()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
  
  time_stamp = str(datetime.now())
  filename = "MV_Dataset_" + time_stamp
  train_log = open("log/"+ filename +"_train.log", "w")
  val_log = open("log/"+ filename +"_val.log", "w")
  test_log = open("log/"+ filename +"_test.log", "w")
  
  train_loader = torch.utils.data.DataLoader(dataset= train_set, batch_size=batch_size, shuffle=True)
  global_step = 0
  for ep in tqdm(range(epochs), desc = ' Epoch Progress:', ncols=900):
    train_iterator = tqdm(train_loader, desc = 'Train Iteration for epoch:'+ str(ep+1), ncols=900)    
    for step, inputs in enumerate(train_iterator):
      model.train()
      optimizer.zero_grad()

      global_step +=1
      x = inputs[0]
      y = inputs[1]
      x = x.to(device)
      y = y.to(device)

      logits = model(x)
      loss = criterion(logits, y)
      loss.backward()
      optimizer.step()

      
    val_loss = evaluate(model, val_set, batch_size, criterion, ep)
    val_log.write("Epoch = {}, validation loss =  {} \n".format(ep+1, val_loss))
    
    if not skip_train_set:
      train_loss  = evaluate(model, train_set, batch_size, criterion, ep)
      train_log.write("Epoch = {}, training loss =  {} \n".format(ep+1, train_loss))
      print("Step = %d, training loss =  %f" %(global_step, train_loss))

    print("Step = %d, validation loss =  %f" %(global_step, val_loss))
    
  test_loss = evaluate(model, test_set, batch_size, criterion, ep)
  test_log.write("End of training, test loss =  {}\n".format(test_loss))
  print("End of Training, test loss =  %f" %(test_loss))

  train_log.close()
  val_log.close()
  test_log.close()

def main():
## main
  input_dim =  10
  output_classes = 1
  learning_rate = 0.001
  batch_size = 16
  epochs = 20
  eval_steps = 100
   ####
  model_dir = 'model_artifacts'
  model_simple_name = 'mv_simple.pt'
  model_complex_name = 'mv_complex.pt'
  ####
  mv_dataset = MVDataset()
  train_set, val_set, test_set = get_get_train_val_test(mv_dataset, 
                                                        val_split=0.40)

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
                                 layers = [50,100,100,50],
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
  
if __name__ == "__main__":
  main()