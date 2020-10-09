import time
import torch
from model.dnn import DenseNeuralNet
from data.churn_data import ChurnDataset
import torch.nn as nn
from utils.util_functions import *
from tqdm.auto import trange, tqdm
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
  
  
  train_log = open("log/train.log", "w")
  val_log = open("log/val.log", "w")
  test_log = open("log/test.log", "w")
  

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

def main():
## main
  input_dim =  13
  output_classes = 2
  learning_rate = 0.001
  batch_size = 16
  epochs = 5
  eval_steps = 100
  ####

  churn_dataset = ChurnDataset()
  train_set, val_set, test_set = get_get_train_val_test(churn_dataset, val_split=0.40)


  model = DenseNeuralNet(input_dim, output_classes)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  train(model, train_set, val_set, test_set , batch_size = batch_size, learning_rate = learning_rate, epochs = epochs, eval_steps = eval_steps)




if __name__ == "__main__":
  main()

