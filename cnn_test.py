import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("qtAgg")

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./cnn_data', train=True, download=True, transform=transform)

test_data = datasets.MNIST(root='./cnn_data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

#Convolutional Layers.  (input, output, kernal size, stride length)
conv1 = nn.Conv2d(1, 6, 3, 1)
conv2 = nn.Conv2d(6, 16, 3, 1)

#Grab a single MNIST image
for i, (X_Train, y_train) in enumerate(train_data):
    break

print(X_Train.shape)

#4D batch
#(one batch, one image, 28, 28)
x = X_Train.view(1,1,28,28)

#First convolution.  Activation function.  Relu unit for activation function.
x = F.relu(conv1(x))

#outputs (1,6,26,26) 26x26 because image is padded.  outer layer is thrown out.  Happened because we didnt explicitly set pad.  But for these images thats fine.
#(1 single image, 6 filters we asked for)
print(x.shape)

#Pooling layer.  Kernal of 2 and stride of 2.
x = F.max_pool2d(x, 2, 2)

#outputs (1,6,13,13)
print(x.shape)

#Second convolutional layer
x = F.relu(conv2(x))

#outputs(1,16,11,11)  padding thing again.
print(x.shape)

#Second pooling

x = F.max_pool2d(x, 2, 2)

#outputs(1,16,5,5)  Rounds down to 5.
print(x.shape)
