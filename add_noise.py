import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import argparse

from model.model import *
import os

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
device = 'cuda'


def main(args):
    batch_size = 64
    if args.model == "MLP":
        if args.dataset == "CIFAR10":
            model = MLP(32*32*3, 10).to(device)
        else:
            # args.dataset == "MNIST"
            model = MLP(28*28, 10).to(device)
    else:
        # elif args.model == "LENET":
        if args.dataset == "CIFAR10":
            model = LeNet_CIFAR10(input_channels=3, output_dim=10).to(device)
        else:
            # args.dataset == "MNIST"
            model = LeNet_MNIST(input_channels=1, output_dim=10).to(device)

    if args.dataset == "CIFAR10":
        # Load CIFAR-10 dataset
        cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # Data loaders
        train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

    elif args.dataset == "MNIST":
        # Load MNIST dataset, loaders
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)


    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        print(model.get_coverage(data, 10))
        print(model.get_coverage(data, 1))
        print(model.get_coverage(data, 0.5))
        exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", "--dataset", choices=["MNIST", "CIFAR10"], default="MNIST")
    parser.add_argument("-m", "--model", choices=["MLP", "CIFAR10"], default="MLP")

    parser.add_argument(
        "--model-path",
        required=True,
        help="save path of model weight",
    )
    args = parser.parse_args()

    main(args)
