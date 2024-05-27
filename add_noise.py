import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from data.utils import *
from model.utils import *


def x2w_mnist(x):
    '''
    Transform pixel value into continuous space -inf ~ inf
    The algorithm is originated from [Nicholas Carlini, David Wagner 
    (2017) Towards Evaluating the Robustness of Neural Networks]
    '''
    # x : [-1, 1],   w : [-inf, inf]
    w = torch.tanh(x)
    return w

def w2x_mnist(w):
    x = torch.atanh(w)
    return x

def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 64
    
    # load original dataset
    train_loader, test_loader = load_dataset(args.dataset, batch_size, shuffle=False)
 
    # load trained model
    model = make_model(args.model, args.dataset).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    w = []
    for data, _ in train_loader:
        data = data.to(device)
        w.append(x2w_mnist(data))

    # w = [60000, 1, 28, 28]
    w = torch.cat(w, 0)
    print(w.shape)
    # optimizer for w
    optimizer = optim.Adam([w], lr=1e-2)

    for data, _ in train_loader:
        data = data.to(device)
        tensor2png(data[1])
        print(model.get_coverage(data, 1))
        exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["MNIST", "CIFAR10"], default="MNIST")
    parser.add_argument("-m", "--model", choices=["MLP", "CIFAR10"], default="MLP")

    parser.add_argument(
        "--model-path",
        required=True,
        help="save path of model weight",
    )
    args = parser.parse_args()

    main(args)
