import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import argparse

from model import *
import os

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n')

    return test_loss

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch

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
        cifar10_train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)
        cifar10_test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print("Training on CIFAR-10")

        # best_valid_loss = float('inf')
        # for epoch in range(1, epochs + 1):
        #     train(model, device, cifar10_train_loader, optimizer, epoch)
        #     valid_loss = test(model, device, cifar10_test_loader)
        #     if valid_loss < best_valid_loss:
        #             best_valid_loss = valid_loss
        #             torch.save(model.state_dict(), 'tut1-model.pt')

        for epoch in range(1, epochs + 1):
            train(model, device, cifar10_train_loader, optimizer, epoch)
            test(model, device, cifar10_test_loader)
    elif args.dagaset == "MNIST":
        # Load MNIST dataset, loaders
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
        mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
        
        # Train and evaluate on MNIST
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        print("Training on MNIST")
        for epoch in range(1, epochs + 1):
            train(model, device, mnist_train_loader, optimizer, epoch)
            test(model, device, mnist_test_loader)

    save_path = os.path.join(args.save_path, f"{args.model}_{args.dataset}.pt")
    torch.save(model.state_dict(), save_path)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", "--dataset", choices=["MNIST", "CIFAR10"], default="MNIST")
    parser.add_argument("-m", "--model", choices=["MLP", "CIFAR10"], default="MNIST")
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-b", "--batch", type=int, default=64)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.001)
    
    parser.add_argument(
        "--save-path",
        required=True,
        help="save path of model weight",
        default="./weights/"
    )

    # parser.add_argument('--iter', type=int, default=100)
    # parser.add_argument("--use-BPDA", action='store_true', help="Enable BPDA")
    # parser.add_argument("--use-dag", action='store_true', help="Enable dag")
    # parser.add_argument("--use-black", action='store_true', help="Enable black-box attack")


    args = parser.parse_args()

    main(args)
