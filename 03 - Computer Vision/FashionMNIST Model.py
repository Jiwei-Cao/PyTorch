import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchmetrics import Accuracy
import numpy as np
import random
import matplotlib.pyplot as plt

# Download FashionMNIST train and test datasets
fashion_mnist_train = datasets.FashionMNIST(root=".",
                                          download=True,
                                          train=True,
                                          transform=transforms.ToTensor())

fashion_mnist_test = datasets.FashionMNIST(root=".",
                                           download=True,
                                           train=False,
                                           transform=transforms.ToTensor())

fashion_mnist_class_names = fashion_mnist_train.classes

# Turn FashionMNIST into dataloaders
fashion_mnist_train_dataloader = DataLoader(fashion_mnist_train,
                                            batch_size=32,
                                            shuffle=True)

fashion_mnist_test_dataloader = DataLoader(fashion_mnist_test,
                                           batch_size=32,
                                           shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"


# Creating the model
class MNIST_model(torch.nn.Module):
    """Model capabale of predicting on MNIST dataset"""
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # Reduces the size of the featured map to highlight most important features
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # Reduces the size of the featured map to highlight most important features
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*7*7,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

model = MNIST_model(input_shape=1,
                    hidden_units=10,
                    output_shape=10).to(device)

# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

acc_fn = Accuracy(task = "multiclass", num_classes=len(fashion_mnist_class_names)).to(device)

# Setup training and testing loop
epochs = 5
for epoch in tqdm(range(epochs)):
    train_loss, test_loss_total = 0, 0
    train_acc, test_acc = 0, 0

    # Training
    model.train()
    for batch, (X_train, y_train) in enumerate(fashion_mnist_train_dataloader):
        X_train, y_train = X_train.to(device), y_train.to(device)

        # Forward pass and loss
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        train_loss += loss
        train_acc += acc_fn(y_pred, y_train)

        # Backpropagation and gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Adjust the loss and acc per epoch
    train_loss /= len(fashion_mnist_train_dataloader)
    train_acc /= len(fashion_mnist_train_dataloader)

    # Testing 
    model.eval()
    with torch.inference_mode():
        for batch, (X_test, y_test) in enumerate(fashion_mnist_test_dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)

            # Foward pass and loss
            y_pred_test = model(X_test)
            test_loss = loss_fn(y_pred_test, y_test)
            test_loss_total += test_loss
            test_acc += acc_fn(y_pred_test, y_test)

        # Adjust the loss and acc per epoch
        test_loss /= len(fashion_mnist_test_dataloader)
        test_acc /= len(fashion_mnist_test_dataloader)
    
    print(f"Epoch: {epoch} | Train loss: {train_loss:.3f} | Train acc: {train_acc:.2f} | Test loss: {test_loss_total:.3f} | Test acc: {test_acc:.2f}")

# Make predictions with the trained model
test_preds = []
model.eval()
with torch.inference_mode():
    for X_test, y_test in tqdm(fashion_mnist_test_dataloader):
        y_logits = model(X_test.to(device))
        y_pred_probs = torch.softmax(y_logits, dim=1)
        y_pred_labels = torch.argmax(y_pred_probs, dim=1)
        test_preds.append(y_pred_labels)
test_preds = torch.cat(test_preds).cpu()

# Get wrong prediction indexes
wrong_pred_indexes = np.where(test_preds != fashion_mnist_test.targets)[0]

# Select 9 random wrong predictions and plot them
random_selection = random.sample(list(wrong_pred_indexes), k=9)

plt.figure(figsize=(10, 10))
for i, idx in enumerate(random_selection):
    # Get true and pred labels
    true_label = fashion_mnist_class_names[fashion_mnist_test[idx][1]]
    pred_label = fashion_mnist_class_names[test_preds[idx]]

    # Plot the wrong prediction with its original label
    plt.subplot(3,3, i+1)
    plt.imshow(fashion_mnist_test[idx][0].squeeze(), cmap="gray")
    plt.title(f"True: {true_label} | Pred: {pred_label}", c="r")
    plt.axis=False