from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# Download FashionMNIST train and test datasets
fashion_mnist_train = datasets.FashionMNIST(root=".",
                                          download=True,
                                          train=True,
                                          transforms=transforms.ToTensor())

fashion_mnist_test = datasets.FashionMNIST(root=".",
                                           download=True,
                                           train=False,
                                           transforms=transforms.ToTensor())

fashion_mnist_class_names = fashion_mnist_train.classes

# Turn FashionMNIST into dataloaders
fashion_mnist_train_dataloader = DataLoader(fashion_mnist_train,
                                            batch_size=32,
                                            shuffle=True)

fashion_mnist_test_dataloader = DataLoader(fashion_mnist_test,
                                           batch_size=32,
                                           shuffle=False)

