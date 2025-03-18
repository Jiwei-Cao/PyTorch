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

# plt.imshow(image.squeeze(), cmap = "gray")
# plt.title(label)
# plt.axis(False)
# plt.show()

torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows, cols = 4, 4
for i in range(1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap = "gray")
    plt.title(class_names[label])
    plt.axis(False)

plt.show()