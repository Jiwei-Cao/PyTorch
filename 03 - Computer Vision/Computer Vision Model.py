import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load MNIST dataset
train_data = datasets.MNIST(root=".",
                           train=True,
                           download=True,
                           transform=transforms.ToTensor())

test_data = datasets.MNIST(root=".",
                           train=False,
                           download=True,
                           transform=transforms.ToTensor())

img = train_data[0][0]
label = train_data[0][1]

class_names = train_data.classes

# Visualising MNIST training dataset
for i in range(5):
    img = train_data[i][0]
    img_squeeze = img.squeeze() # Squeezing to drop the extra dimsension so that matplotlib can use the data with the correct shape
    label = train_data[i][1]
    plt.figure(figsize=(3,3))
    plt.imshow(img_squeeze, cmap="gray")
    plt.title(label)
    plt.axis(False)

# Create train dataloader (use a dataloader to split dataset into mini-batches for training)
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=32,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=32,
                             shuffle=False)

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