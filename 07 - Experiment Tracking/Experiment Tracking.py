import torch
import torchvision

print(torch.__version__)
print(torchvision.__version__)

# Regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

# Try to get torchinfo, install it if it doesn't work
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    # !pip install -q torchinfo
    from torchinfo import summary

# Try to import the going_modular directory, download it from GitHub if it doesn't work
try:
    from going_modular import data_setup, engine
except:
    # Get the going_modular scripts
    print("[INFO] Couldn't find going_modular scripts... downloading them from GitHub.")
    # !git clone https://github.com/Jiwei-Cao/PyTorch
    # !mv "PyTorch/05 - Going Modular/going_modular" .
    # !rm -rf PyTorch
    from going_modular import data_setup, engine

device = "cuda" if torch.cuda.is_available() else "cpu"
device

def set_seeds(seed: int = 42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

set_seeds()

# Get Data

import os
import zipfile

from pathlib import Path

import requests

def download_data(source: str,
                  destination: str, 
                  remove_source: bool = True) -> Path:
  """Downloads a zipped dataset from source and unzips to destination"""

  # Setup path to folder
  data_path = Path("data/")
  image_path = data_path / destination

  # If the image folder doesn't exist, download and prepare it
  if image_path.is_dir():
    print(f"[INFO] {image_path} directory already exists, skipping download")
  else:
    print(f"[INFO] Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)

  # Download the target data
  target_file = Path(source).name
  with open(data_path / target_file, "wb") as f:
      request = requests.get(source)
      print(f"[INFO] Downlaoding {target_file} from {source}...")
      f.write(request.content)

  # Unzip target file
  with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
    print(f"[INFO] Unzipping {target_file} data...")
    zip_ref.extractall(image_path)

  # Remove .zip file if needed
  if remove_source:
    os.remove(data_path / target_file)

  return image_path

image_path = download_data(source="https://github.com/Jiwei-Cao/PyTorch/raw/refs/heads/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")

# Setup directories
train_dir = image_path / "train"
test_dir = image_path / "test"

train_dir, test_dir

# Create Datasets and DataLoaders

# Manual Creation
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

manual_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transform,
    batch_size=32
)

train_dataloader, test_dataloader, class_names

# Auto creation

# Setup pretrinaed weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
weights

# Get transforms from weights
auto_transforms = weights.transforms()
auto_transforms

# Create DataLoaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform = auto_transforms,
    batch_size=32
)

train_dataloader, test_dataloader, class_names

# Setting up a pretrained model

model = torchvision.models.efficientnet_b0(weights=weights).to(device)
# model

# Freeze all base layers
for param in model.features.parameters():
  param.requires_grad = False

# Adjust the classifier head
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, out_features=len(class_names))
).to(device)

summary(
    model=model,
    input_size=(32, 3, 224, 224),
    verbose=0,
    col_names=["input_size", "output_size", "num_params", "trainable"],
    col_width=20,
    row_settings=["var_names"]
)