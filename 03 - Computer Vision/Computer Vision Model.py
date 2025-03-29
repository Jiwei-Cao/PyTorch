import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as pyt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load MNIST data
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