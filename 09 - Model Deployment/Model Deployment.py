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

# Training the ViT feature extractor
optimizer = torch.optim.Adam(params=vit.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

# Train ViT feature extractor 
set_seeds()
vit_results = engine.train(model=vit,
                           train_dataloader=train_dataloader_vit,
                           test_dataloader=test_dataloader_vit,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=10,
                           device=device)

# Plot loss curves of the ViT feature extractor
plot_loss_curves(vit_results)

# Saving ViT feature extractor
utils.save_model(model=vit,
                 target_dir="models",
                 model_name="09_pretrained_vit_feature_extractor_pizza_steak_sushi_20_percent.pth")

# Checking the size of the ViT feature extractor
pretrained_vit_model_size = Path("models/09_pretrained_vit_feature_extractor_pizza_steak_sushi_20_percent.pth").stat().st_size / (1024 * 1024)
print(f"Pretrained ViT feature extractor model size: {round(pretrained_vit_model_size, 2)}MB")

# Collecting ViT feature extractor stats

# Count number of parameters in Vit
vit_total_params = sum(torch.numel(param) for param in vit.parameters())
vit_total_params

# Create ViT statistics dictionary
vit_stats = {"test_loss": vit_results["test_loss"][-1],
             "test_acc": vit_results["test_acc"][-1],
             "number_of_parameters": vit_total_params,
             "model_size (MB)": pretrained_vit_model_size}
vit_stats

# Making predictions with the trained models and timing them

# Get all test data paths
test_data_paths = list(Path(test_dir).glob("*/*.jpg"))
test_data_paths[:5]

# Creating a function to make predictions across the test dataset
import pathlib
from PIL import Image
from timeit import default_timer as timer
from tqdm.auto import tqdm
from typing import List, Dict

def pred_and_store(paths: List[pathlib.Path],
                   model: torch.nn.Module,
                   transform: torchvision.transforms,
                   class_names: List[str],
                   device: str = "cuda" if torch.cuda.is_available() else "cpu") -> List[Dict]:
  # Create an empty list
  pred_list = []

  # Loop through the target input paths
  for path in tqdm(paths):
    # Create an empty dictionary
    pred_dict = {}

    # Get the sample path and ground truth class from the filepath
    pred_dict["image_path"] = path
    class_name = path.parent.stem
    pred_dict["class_name"] = class_name

    # Start the prediction timer
    start_time = timer()

    # Open the image using Image.open(path)
    img = Image.open(path)

    # Transform the image to be usable with a given model
    transformed_image = transform(img).unsqueeze(0).to(device)

    # Prepare the model for inference by sending to the target device
    model = model.to(device)
    model.eval()

    # Calculate prediction probabilities and prediction class
    pred_logit = model(transformed_image)
    pred_prob = torch.softmax(pred_logit, dim=1)
    pred_label = torch.argmax(pred_prob, dim=1)
    pred_class = class_names[pred_label.cpu()]

    # Add the prediction probability and prediction class to empty dictionary 
    pred_dict["pred_prob"] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
    pred_dict["pred_class"] = pred_class

    # End the prediction timer and add the time to the prediction dictionary
    end_time = timer()
    pred_dict["time_for_pred"] = round(end_time-start_time, 4)

    # See if the predicted class matches the ground truth class
    pred_dict["correct"] = class_name == pred_class

    # Append the updated prediction dictionary to the empty list of predictions
    pred_list.append(pred_dict)

  return pred_list

# Making and timing predictions with EffNetB2
effnetb2_test_pred_dicts = pred_and_store(paths=test_data_paths,
                                          model=effnetb2,
                                          transform=effnetb2_transforms,
                                          class_names=class_names,
                                          device="cpu")

effnetb2_test_pred_dicts

# Turn the test_pred_dicts into a dataframe
import pandas as pd
effnetb2_test_pred_df = pd.DataFrame(effnetb2_test_pred_dicts)
effnetb2_test_pred_df.head()

# Check number of correct predictions
effnetb2_test_pred_df.correct.value_counts()

# Find the average time per prediction
effnetb2_average_time_per_pred = round(effnetb2_test_pred_df.time_for_pred.mean(), 4)
print(f"EffNetB2 average time per prediction: {effnetb2_average_time_per_pred}")

# Add time per prediction to EffNetB2 stats dictionary
effnetb2_stats["time_per_pred_cpu"] = float(effnetb2_average_time_per_pred)
effnetb2_stats

# Making and timing predictions with ViT
vit_test_pred_dicts = pred_and_store(paths=test_data_paths,
                                     model=vit,
                                     transform=vit_transforms,
                                     class_names=class_names,
                                     device="cpu")

vit_test_pred_dicts

vit_test_pred_df = pd.DataFrame(vit_test_pred_dicts)
vit_test_pred_df.head()

# Check number of correct predictions
vit_test_pred_df.correct.value_counts()

# Calculate average time per prediction for ViT model
vit_average_time_per_pred = round(vit_test_pred_df.time_for_pred.mean(), 4)
print(f"ViT average time per prediction: {vit_average_time_per_pred}")

# Add average time per prediction to ViT stats
vit_stats["time_per_pred_cpu"] = float(vit_average_time_per_pred)
vit_stats

# Comparing model results, prediction times and size

# Turn stat dictionaries into DataFrame
df = pd.DataFrame([effnetb2_stats, vit_stats])

# Add column for model names
df["model"] = ["EffNetB2", "ViT"]

# Convert accuracy to percentages
df["test_acc"] = round(df["test_acc"] * 100, 2)

df

# Compare ViT to EffNetB2 across different characteristics
pd.DataFrame(data=(df.set_index("model").loc["ViT"]) / df.set_index("model").loc["EffNetB2"],
             columns=["ViT to EffNetB2 ratios"]).T

# Visualising the speed vs performance tradeoff

# Create a plot from model comparison DataFrame
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(data=df,
                     x="time_per_pred_cpu",
                     y="test_acc",
                     c=["blue", "orange"],
                     s="model_size (MB)")

# Add titles and labels 
ax.set_title("FoodVision Mini Inference Speed vs Performance", fontsize=18)
ax.set_xlabel("Prediction time per image (seconds)", fontsize=14)
ax.set_ylabel("Test_accuracy (%)", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.grid(True)

# Annotate the samples on the scatter plot 
for index, row in df.iterrows():
  ax.annotate(text=row["model"],
              xy=(row["time_per_pred_cpu"]+0.0006, row["test_acc"]+0.03),
              size=12)
  
# Create a legend based on the model sizes 
handles, labels = scatter.legend_elements(prop="sizes", alpha=0.5)
model_size_legend = ax.legend(handles,
                              labels,
                              loc="lower right",
                              title="Model size (MB)",
                              fontsize=12)

# Save the figure
plt.savefig("09-foodvision-mini-inference-speed-vs-performance.png")