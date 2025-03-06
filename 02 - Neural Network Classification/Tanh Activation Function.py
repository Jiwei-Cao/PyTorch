import torch

tensor_A = torch.arange(-100, 100, 1)
plt.plot(tensor_A)

plt.plot(torch.tanh(tensor_A))

def tanh(x):
    return ((torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x)))
plt.plot(tanh(tensor_A))