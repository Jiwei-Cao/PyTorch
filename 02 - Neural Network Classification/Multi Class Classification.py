# Code for creating a spiral dataset from CS231n
import numpy as np
import matplotlib.pyplot as plt
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j
# lets visualize the data
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
plt.show()

# Turn data into tensors
import torch
import torch.nn as nn

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.LongTensor)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = RANDOM_SEED)

device = "cuda" if torch.is_available() else "cpu"

from torchmetrics import Accuracy
acc_fn = Accuracy(task="multiclass", num_classes=3).to(device)

class SpiralModel(nn.Module):
    def __init__(self):
       super().__init__()
       self.linear1 = nn.Linear(in_features = 2, out_features = 10)
       self.linear2 = nn.Linear(in_features = 10, out_features = 10)
       self.linear3 = nn.Linear(in_features = 10, out_features = 3)
       self.relu = nn.ReLU()

    def forward(self, x):
       return self.linear3(self.relu(self.linear2(self.relu(self.linear1(x)))))
    
model_1 = SpiralModel().to(device)

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_1.parameters(), 
                             lr = 0.02)

epochs = 1000

for epoch in range(epochs):
   model_1.train()

   y_logits = model_1(X_train)
   y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

   loss = loss_fn(y_logits, y_train)
   acc = acc_fn(y_pred, y_train)

   optimizer.zero_grad()
   loss.backward()
   optimizer.step()