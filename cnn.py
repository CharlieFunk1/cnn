import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import time
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

#Model Class
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #Convolutional layers
        self.conv1 = nn.Conv2d(1,6,3,1)
        self.conv2 = nn.Conv2d(6,16,3,1)
        #Fully connected layers. (neurons)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    #Forward pass.  2 Convvolutional layers and 2 pooling layers.
    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        #Re-view to flatten it out
        X = X.view(-1, 16*5*5)
        #Fully connected layers
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

torch.manual_seed(41)
model = ConvolutionalNetwork()
print(model)

#if torch.cuda.is_available():
#    model.cuda()
#model = model.cuda()
#device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
#device = 'cuda'
#model = model.to(device)
#device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using device: {device}")
#for data in train_loader:
#    inputs, labels = data
#
#    inputs = inputs.to(device)
#
#    labels = labels.to(device)
#
#    outputs = model(inputs)
#
#for data in test_loader:
#    inputs, labels = data
#
#    inputs = inputs.to(device)
#
#    labels = labels.to(device)
#    
#    outputs = model(inputs)




        
#Loss Function Optimizer
criterion = nn.CrossEntropyLoss()
#optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#start timer to see how long this takes
start_time = time.time()

#Create variables to keep track of things
epochs = 5
train_losses=[]
test_losses=[]
train_correct=[]
test_correct=[]

#For Loop of Epochs
for i in range(epochs):
    trn_corr = 0
    tst_corr = 0
    #Train
    for b,(X_train, y_train) in enumerate(train_loader):
        b += 1 # start our batches at 1
        y_pred = model(X_train) #Get predicted values from training set.  Not Flattened.  It's 2d.
        loss = criterion(y_pred, y_train)#How off are we?  Compare the predictionsto correct values
        
        #add up the number of correct predictions.
        predicted = torch.max(y_pred.data, 1)[1]
        
        #how many we got correct from this batch.  True = 1 False = 0.  We sum it up.
        batch_corr = (predicted == y_train).sum()

        #Keep track as we go along in training.
        trn_corr += batch_corr

        #Update our parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #Print out results
        if b%600 == 0:
            print(f'Epoch: {i}  Batch: {b}  Loss: {loss.item()}')

    train_losses.append(loss)
    train_correct.append(trn_corr)

    #Test
    #No gradient so we dont mess up our weights and bias with test
    with torch.no_grad():
        for b,(X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1] #adding up correct predictions
            tst_corr += (predicted == y_test).sum() #T=1 F=0 and sum up

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)
        

current_time = time.time()
total = current_time - start_time
print(f'Training Took: {total/60} minutes!')

train_losses = [tl.item() for tl in train_losses]
plt.plot(train_losses, label="Training Loss")
plt.plot(test_losses, label="Validation Loss")
plt.title("Loss at Epoch")
plt.legend()
plt.show()
plt.plot([t/600 for t in train_correct], label="Training accuracy")
plt.plot([t/100 for t in test_correct], label="Validation Accuracy")
plt.title("Accuracy at the end of each epoch")
plt.legend()
plt.show()

