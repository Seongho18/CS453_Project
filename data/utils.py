from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_dataset(dataset, batch_size, shuffle=True):
    if dataset == "CIFAR10":
        # Load CIFAR-10 dataset
        cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        cifar10_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # Data loaders
        train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(cifar10_test, batch_size=batch_size, shuffle=shuffle)

    elif dataset == "MNIST":
        # Load MNIST dataset
        mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        # Data loaders
        train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


def save_png(img_tensor, name):
    transform = transforms.ToPILImage()
    img = transform(img_tensor)
    img.save(f'{name}.png', 'png')



def tensor2png(img_tensor):
    transform = transforms.ToPILImage()
    img = transform(img_tensor)
    img.save('sample.png', 'png')
    return img
