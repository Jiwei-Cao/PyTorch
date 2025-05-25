import torch

#2
x = torch.rand(size=(7,7))
# print(x)

#3
y = torch.rand(size=(1,7))
z = torch.matmul(x, y.T)
# print(z)

#4
torch.manual_seed(0)
x = torch.rand(size=(7,7))
y = torch.rand(size=(1,7))
z = torch.matmul(x, y.T)
# print(z, z.shape)

#5
torch.cuda.manual_seed(1234)

#6
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Device: {device}")
x = torch.rand(size=(2,3)).to(device)
y = torch.rand(size=(2,3)).to(device)
# print(x, y)

#7
z = torch.matmul(x,y.T)
# print(z)

#8
max = torch.max(z)
min = torch.min(z)
# print(max, min)

#9 
arg_max = torch.argmax(z)
arg_min = torch.argmin(z)
# print(arg_max, arg_min)

#10
torch.manual_seed(7)
random_tensor = torch.rand(size=(1,1,1,10))
squeezed_random_tensor = random_tensor.squeeze()
print(random_tensor, random_tensor.shape)
print(squeezed_random_tensor, squeezed_random_tensor.shape)