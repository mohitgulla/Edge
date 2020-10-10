# Note : Order of layers should always be : Conv -> Normalization -> Activation -> Dropout -> Pooling
import torch
import torch.nn as nn
class DenseNeuralNet(nn.Module):
  """ 
  Class to create the architecture of dense neural network and the forward pass of batch 
  """
  def __init__(self, input_size, num_classes):
    super().__init__()
    config = {
        "num_layers" : 2,
        "nodes_per_layer" : [20, num_classes],
        "dropout_prob" : 0.1,
        "batch_norm" : False
    }

    layers = []
    prev = input_size
  
    for i in range(0, config["num_layers"]):
      layers.append(nn.Linear(prev, config["nodes_per_layer"][i]))
      if not i==config["num_layers"]-1:
        if config["batch_norm"]:
          layers.append(nn.BatchNorm1d(num_features = config["nodes_per_layer"][i]))
        layers.append(nn.ReLU())

        if config["dropout_prob"] > 0:
          layers.append(nn.Dropout(p = config["dropout_prob"]))

      prev = config["nodes_per_layer"][i]
    
    self.net = nn.Sequential(*layers)

    
  def forward(self, x):
      out = self.net(x)
      return out