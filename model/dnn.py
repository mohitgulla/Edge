# Note : Order of layers should always be : Conv -> Normalization -> Activation -> Dropout -> Pooling
import torch
import torch.nn as nn
class DenseNeuralNet(nn.Module):
  """
  Class to create the architecture of dense neural network and the forward pass of batch
  """
  def __init__(self, input_size, num_classes,layers=[],dropout_prob = 0.0, batch_norm = False):
    super().__init__()
    layers.append(num_classes)
    config = {
        "num_layers" : len(layers),
        "nodes_per_layer" : layers,
        "dropout_prob" : dropout_prob,
        "batch_norm" : batch_norm
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
