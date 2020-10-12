# Note : Order of layers should always be : Conv -> Normalization -> Activation -> Dropout -> Pooling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, resnet50

# Regular Convolution model for 3,32,32 images
class CNN(nn.Module):
  """CNN."""
  def __init__(self, num_classes):
    super(CNN, self).__init__()
    self.network = nn.Sequential(
        # block 1
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # output: 64 x 16 x 16
        
        # block 2
        nn.Dropout(0.2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # output: 128 x 8 x 8
        
        # block 3
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # output: 256 x 4 x 4
        
        # block 4 - FC layers
        nn.Dropout(0.3),
        nn.Flatten(), 
        nn.Linear(256*4*4, 512),
        nn.ReLU(),
        #nn.Linear(1024, 512),
        #nn.ReLU(),
        nn.Linear(512, num_classes))


  def forward(self, x):
    return self.network(x)


# ResNet9 Model
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class CNN_resnet9(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(3, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Linear(512, num_classes))
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out


# ResNet50 using pre-trained model
class CNN_resnet50(nn.Module):
  def __init__(self, n_classes):
    super(CNN_resnet50, self).__init__()
    self.network = resnet50(pretrained=True)
    for param in self.network.parameters():
      param.requires_grad = False

    self.network.fc = nn.Sequential(
                          nn.Linear(2048, 1024), 
                          nn.ReLU(), 
                          nn.Linear(1024, n_classes))
    
  def forward(self, x):
    return self.network(x)


# VGG16 using pre-trained model
class CNN_vgg16(nn.Module):
  def __init__(self, n_classes):
    super(CNN_vgg16, self).__init__()
    self.network = vgg16(pretrained=True)
    for param in self.network.parameters():
      param.requires_grad = False
    self.network.classifier = nn.Sequential(
                          nn.Dropout(0.5),
                          nn.Linear(25088, 4096), 
                          nn.ReLU(), 
                          nn.Dropout(0.4),
                          nn.Linear(4096, 256), 
                          nn.ReLU(), 
                          nn.Linear(256, n_classes),                   
                          nn.Softmax(dim=1))
    
  def forward(self, x):
    return self.network(x)

