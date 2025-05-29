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