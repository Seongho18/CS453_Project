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

    # initialzie noise vector
    noise = torch.randn((batch_size, 1, 28, 28), requires_grad=True, device=device)
    optimizer_noise = optim.Adam([noise], lr=1e-2)

    for data, _ in train_loader:
        # data = data.to(device)
        # print("##")
        # print(data.size)
        # tensor2png(data[1])
        # print(model.get_coverage(data, 1))
        # exit()

        # Optimize noise iteratively
        for _ in range(args.step_num):
            # load data
            data = data.to(device)

            # init optimizer
            optimizer_noise.zero_grad()
            
            # perturbed image
            perturbed_data = data + noise

            # apply constraints
            perturbed_data = torch.clamp(perturbed_data, -1, 1)

            # compute coverage
            coverage = model.get_coverage(perturbed_data, 1)

            # our loss function
            loss_noise = torch.norm(noise) * args.lagrangian - coverage.mean()

            # optimize
            loss_noise.backward()
            optimizer_noise.step()

        tensor2png(perturbed_data[1])
        print(model.get_coverage(perturbed_data, 1))
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
    # args for noise
    parser.add_argument("-t", "--step-num", default=100, type=int)
    parser.add_argument("-l", "--lagrangian", default=0.1, type=float)

    args = parser.parse_args()

    main(args)
