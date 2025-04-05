import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import requests
import zipfile
from pathlib import Path
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Get data
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it
if not image_path.is_dir():
  image_path.mkdir(parents=True, exist_ok=True)

# Download pizza, steak and sushi data images from github
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
  request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
  f.write(request.content)

# Unzip pizza, steak and sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
  zip_ref.extractall(image_path)

# 2. Setting up data
train_dir = image_path / "train"
test_dir = image_path / "test"

# Visualise an image
# 1. Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# 2. Get random image path
random_image_path = random.choice(image_path_list)

# 3. Get image class from path name
image_class = random_image_path.parent.stem

# 4. Open image
img = Image.open(random_image_path)

# Print metadata
# print(f"Random image path: {random_image_path}")
# print(f"Image class: {image_class}")
# print(f"Image height: {img.height}")
# print(f"Image width: {img.width}")
# img

# Visualise an image with matplotlib
# 1. Turn the image into an array
img_as_array = np.asarray(img)

# 2. Plot the image
# plt.figure(figsize=(10, 7))
# plt.imshow(img_as_array)
# plt.title(f"Image class: {image_class} | Image shape: {img_as_array.shape} -> [height, width, color_channels]")
# plt.axis(False)