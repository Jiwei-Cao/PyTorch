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