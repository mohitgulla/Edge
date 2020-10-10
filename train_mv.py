import time
import torch
import torch.nn as nn
from model.dnn import DenseNeuralNet
from data.mv_data import MVDataset
from utils.util_functions import *
from tqdm.auto import trange, tqdm
from tqdm import trange
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


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
  
  train_log = open("log/train.log", "w")
  val_log = open("log/val.log", "w")
  test_log = open("log/test.log", "w")
  

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
  epochs = 10
  eval_steps = 100
  ####

  churn_dataset = MVDataset()
  train_set, val_set, test_set = get_get_train_val_test(churn_dataset, val_split=0.40)


  model = DenseNeuralNet(input_dim, output_classes)



  train(model, train_set, val_set, test_set , batch_size = batch_size, learning_rate = learning_rate, epochs = epochs, eval_steps = eval_steps, skip_train_set = True)


if __name__ == "__main__":
  main()

