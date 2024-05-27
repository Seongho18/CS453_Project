import torch.nn.functional as F

from model.model import *

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

def make_model(model_structure, dataset):
    if model_structure == "MLP":
        if dataset == "CIFAR10":
            model = MLP(32*32*3, 10)
        else:
            # dataset == "MNIST"
            model = MLP(28*28, 10)
    else:
        # elif model_structure == "LENET":
        if dataset == "CIFAR10":
            model = LeNet_CIFAR10(input_channels=3, output_dim=10)
        else:
            # dataset == "MNIST"
            model = LeNet_MNIST(input_channels=1, output_dim=10)
    return model