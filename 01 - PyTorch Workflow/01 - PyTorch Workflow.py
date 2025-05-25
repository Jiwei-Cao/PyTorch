import torch
from torch import nn
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1
# Create the data parameters
weight = 0.3
bias = 0.9
X = torch.arange(0, 1, 0.01).unsqueeze(dim=1)
y = weight * X + bias
# print(f"Number of X samples: {len(X)}")
# print(f"Number of y samples: {len(X)}")
# print(f"First 10 X and y samples:\nX: {X[:10]}\ny:{y[:10]}")

# Split the data into training and testing
train_split = int(len(X) * 0.8)
X_train = X[:train_split]
y_train = y[:train_split]
X_test = X[train_split:]
y_test = y[train_split:]
# print(len(X_train), len(y_train), len(X_test), len(y_test))

# Plot the training and testing data
def plot_predictions(
        train_data = X_train,
        train_labels = y_train,
        test_data = X_test,
        test_labels = y_test,
        predictions = None):
    plt.figure(figsize = (10,7))
    plt.scatter(train_data, train_labels, c = 'b', s = 4, label = "Training data")
    plt.scatter(test_data, test_labels, c = 'g', s = 4, label = "Test data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c = 'r', s = 4, label = "Predictions")
    plt.legend(prop = {"size" : 14})
    plt.show() # running on a python script instead of jupyter notebook 
#plot_predictions()



# 2
# Create PyTorch linear regression model by subclassing nn.Module
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(data=torch.randn(1, requires_grad=True))
        self.bias = nn.Parameter(data=torch.randn(1, requires_grad=True))

    def forward(self, x):
        return self.weight * x + self.bias
    
torch.manual_seed(42)
model_1 = LinearRegressionModel()
# print(model_1, model_1.state_dict())

# Instantiate the model and put it to the target device
model_1.to(device)
# print(list(model_1.parameters()))



# 3
# Create the loss function and the optimizer
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params = model_1.parameters(), lr = 0.01)

# Training loop - train model for 300 epochs
torch.manual_seed(42)
epochs = 300

# Send data to target device
X_train = X_train.to(device)
X_test = X_test.to(device)
Y_train = y_train.to(device)
Y_test = y_test.to(device)

for epoch in range(epochs):
    ### Training

    # Put model in train mode
    model_1.train()

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero gradients
    optimizer.zero_grad()

    # 4. Backpropagation
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()

    ### Perform testing every 20 epochs
    if epoch % 20 == 0:

        # Put model in evaluation mode and setup inference context
        model_1.eval()
        with torch.inference_mode():

            # 1. Forward pass
            y_preds = model_1(X_test)
            
            # 2. Calculate test loss
            test_loss = loss_fn(y_preds, y_test)

            # Print out what's happening
            # print(f"Epoch: {epoch} | Train loss: {loss:.3f} | Test loss: {test_loss:.3f}")



# 4
# Make predictions with the model
model_1.eval()
with torch.inference_mode():
    y_preds = model_1(X_test)
# print(y_preds)

# Plot the predictions
# plot_predictions(predictions = y_preds.cpu())



# 5
from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents = True, exist_ok = True)
# 2. Create model save path
MODEL_NAME = "01 pytorch model"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
# 3. Save the model state dict
print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(obj = model_1.state_dict(), f = MODEL_SAVE_PATH)

# Create new instance of model and load saved state dict 
loaded_model = LinearRegressionModel()
loaded_model.load_state_dict(torch.load(f = MODEL_SAVE_PATH))
loaded_model.to(device)

# Make predictions with the loaded model and compare them to the previous
y_preds_new = loaded_model(X_test)
print(y_preds == y_preds_new)
print(loaded_model.state_dict())