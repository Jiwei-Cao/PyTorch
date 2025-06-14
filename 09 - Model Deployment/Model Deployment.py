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

device = "cuda" if torch.cuda.is_available() else "cpu"
device

# Download pizza, steak and sushi images from github
data_20_percent_path = download_data(source="https://github.com/Jiwei-Cao/PyTorch/raw/refs/heads/main/data/pizza_steak_sushi_20_percent.zip",
                                     destination="pizza_steak_sushi_20_percent")
data_20_percent_path

# Setup train and test paths
train_dir = data_20_percent_path / "train"
test_dir = data_20_percent_path / "test"

train_dir, test_dir

# Creating an EffNetB2 feature extractor

# Pretrained EffNetB2 weights
effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT

# Get EffNetB2 transforms
effnetb2_transforms = effnetb2_weights.transforms()

# Setup pretrained model instance
effnetb2 = torchvision.models.efficientnet_b2(weights=effnetb2_weights)

# Freeze base layers in the model
for param in effnetb2.parameters():
  param.requires_grad = False

# summary(model=effnetb2,
#         input_size=(1, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

effnetb2.classifier

set_seeds()
effnetb2.classifier = nn.Sequential(
    nn.Dropout(p=0.3, inplace=True),
    nn.Linear(in_features=1408, out_features=3)
)

# summary(model=effnetb2,
#         input_size=(1, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

# Creating a function to make an EffNetB2 feature extractor
def create_effnetb2_model(num_classes:int=3,
                          seed:int=42):
  # Create EffNetB2 pretrained weights, transform and model
  weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
  transform = weights.transforms()
  model = torchvision.models.efficientnet_b2(weights=weights)

  # Freeze all layers in the base model
  for param in model.parameters():
    param.requires_grad = False

  # Change classifier head 
  torch.manual_seed(seed)
  model.classifier = nn.Sequential(
      nn.Dropout(p=0.3, inplace=True),
      nn.Linear(in_features=1408, out_features=num_classes)
  )

  return model, transform

effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=3,
                                                      seed=42)

# summary(model=effnetb2,
#         input_size=(1, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

effnetb2_transforms

# Setup dataloaders
train_dataloader_effnetb2, test_dataloader_effnetb2, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                                 test_dir=test_dir,
                                                                                                 transform=effnetb2_transforms,
                                                                                                 batch_size=32)

len(train_dataloader_effnetb2), len(test_dataloader_effnetb2), class_names

# Training the EffNetB2 feature extractor
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=effnetb2.parameters(), 
                             lr=1e-3)
set_seeds()
effnetb2_results = engine.train(model=effnetb2.to(device),
                                train_dataloader=train_dataloader_effnetb2,
                                test_dataloader=test_dataloader_effnetb2,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                epochs=10,
                                device=device)

# Inspecting EffNetB2 loss curves
plot_loss_curves(effnetb2_results)

# Saving EffNetB2 feature extractor
from going_modular import utils

utils.save_model(model=effnetb2,
                 target_dir="models",
                 model_name="09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth")

# Inspecting the size of the EffNetB2 feature extractor
from pathlib import Path

pretrained_effnetb2_model_size = Path("models/09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth").stat().st_size / (1024 * 1024)
print(f"Pretrained EffNetB2 feature extractor model size: {round(pretrained_effnetb2_model_size, 2)}MB")

# Collecting EffNetB2 feature extractor stats

# Count number of parameters in EffNetB2
effnetb2_total_params = sum(torch.numel(param) for param in effnetb2.parameters())
effnetb2_total_params

# Create a dictionary with EffNetB2 statistics
effnetb2_stats = {"test_loss": effnetb2_results["test_loss"][-1],
                  "test_acc": effnetb2_results["test_acc"][-1],
                  "number_of_parameters": effnetb2_total_params,
                  "model_size (MB)": pretrained_effnetb2_model_size}

effnetb2_stats

# Creating a ViT feature extractor
vit = torchvision.models.vit_b_16()
vit.heads

# Creating a function to make a ViT feature extractor
def create_vit_model(num_classes:int=3,
                     seed:int=42):
  # Create a ViT_B_16 pretrained weights, transforms and model
  weights = torchvision.models.ViT_B_16_Weights.DEFAULT
  transforms = weights.transforms()
  model = torchvision.models.vit_b_16(weights=weights)

  # Freeze all of the base layers
  for param in model.parameters():
    param.requires_grad = False

  # Change the classifier head
  torch.manual_seed(seed)
  model.heads = nn.Sequential(
      nn.Linear(in_features=768, out_features=num_classes)
  )

  return model, transforms

vit, vit_transforms = create_vit_model()

# summary(model=vit,
#         input_size=(1, 3, 224, 224),
#         col_names=["input_size", "output_size", "num_params", "trainable"],
#         col_width=20,
#         row_settings=["var_names"])

# Create dataloaders for vit feature extractor
train_dataloader_vit, test_dataloader_vit, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                       test_dir=test_dir,
                                                                                       transform=vit_transforms,
                                                                                       batch_size=32)
len(train_dataloader_vit), len(test_dataloader_vit), class_names