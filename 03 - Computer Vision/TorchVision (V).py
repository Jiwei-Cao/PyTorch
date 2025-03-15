import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# 1. Getting a dataset
train_data = datasets.FashionMNIST(
    root = "data", 
    train = True,
    download =True,
    transform = ToTensor(),
    target_transform = None
)

test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
    target_transform = None
)

class_names = train_data.classes
# print(class_names)

image, label = train_data[0]
# print(f"Image shape: {image.shape}")
plt.imshow(image.squeeze())
plt.show()