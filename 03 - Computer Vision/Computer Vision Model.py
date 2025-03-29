import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt

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