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

# plt.show()

# Turn datasets into iterables (batches)
from torch.utils.data import DataLoader
BATCH_SIZE = 32

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             shuffle=False)

# Check what's inside the training dataloader by turning it into an iterable and getting the next batch
train_features_batch, train_labels_batch = next(iter(train_dataloader))

# Show a sample
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
# print(f"Image size: {img.shape}")
# print(f"Label: {label}, label size: {label.shape}")
# plt.show()

# Building a baseline model

# Create a flatten layer
flatten_model = nn.Flatten()
# Get a single sample
x = train_features_batch[0]
# Flatten the sample
output = flatten_model(x)

# print(f"Shape before flattening: {x.shape}") [colour_channels, height, width]
# print(f"Shape after flattening: {output.shape}") [colour_channels, height*width]

# Creating a model
class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().init()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )
    
    def forward(self, x):
        return self.layer_stack(x)
    
torch.manual_seed(42)
model_0 = FashionMNISTModelV0(
    input_shape=784, # 28*28
    hidden_units=10,
    output_shape=len(class_names) # one output value for every class
)
