import torch
import torchvision
from torchvision import datasets
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

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
