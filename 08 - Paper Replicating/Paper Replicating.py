import torch
import torchvision

# Regular imports
import matplotlib.pyplot as plt

from torch import nn
from torchvision import transforms

try:
  from torchinfo import summary
except:
  print("[INFO] Couldn't find torchinfo... installing it")
#   !pip install -q torchinfo
  from torchinfo import summary

# Import going_modular directory
try:
  from going_modular import data_setup, engine
  from helper_functions import download_data, set_seeds, plot_loss_curves
except:
  # Get the going_modular scripts
  print("[INFO] Couldn't find going_modular scripts... downloading them from github")
#   !git clone https://github.com/Jiwei-Cao/PyTorch
#   !mv "PyTorch/05 - Going Modular/going_modular" .
#   !mv "PyTorch/helper_functions.py" .
#   !rm -rf PyTorch
  from going_modular import data_setup, engine
  from helper_functions import download_data, set_seeds, plot_loss_curves

# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
device

image_path = download_data(source="https://github.com/Jiwei-Cao/PyTorch/raw/refs/heads/main/data/pizza_steak_sushi.zip",
                        destination="pizza_steak_sushi")
image_path

train_dir = image_path / "train"
test_dir = image_path / "test"

train_dir, test_dir

# Create image size 
IMG_SIZE = 224 # comes from table 3 of the ViT paper

# Create transforms pipeline
manual_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Create dataloaders
BATCH_SIZE = 32

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms,
    batch_size=BATCH_SIZE
)

len(train_dataloader), len(test_dataloader), class_names

# Get a batch of images
image_batch, label_batch = next(iter(train_dataloader))

# Get a single image and label from the batch
image, label = image_batch[0], label_batch[0]

# View the single image and label shapes
image.shape, label

# Plot the image with matplotlib
plt.imshow(image.permute(1, 2, 0))
plt.title(class_names[label])
_ = plt.axis(False)

# Create example values
height = 224 
width = 224
color_channels = 3
patch_size = 16

# Calculate the number of patches
number_of_patches = int((height * width) / patch_size**2)
number_of_patches

# Input shape 
embedding_layer_input_shape = (height, width, color_channels)

# Output shape
embedding_layer_output_shape = (number_of_patches, patch_size**2 * color_channels)

print(f"Input shape (single 2D image): {embedding_layer_input_shape}")
print(f"Ouptut shape (single 1D sequence of patches): {embedding_layer_output_shape}")

# View a single image
plt.imshow(image.permute(1, 2, 0))
plt.title(class_names[label])
_ = plt.axis(False)