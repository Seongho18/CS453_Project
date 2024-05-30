import argparse
import os
import shutil
import cv2

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

    # Create directories for saving images
    if os.path.exists(args.original_data_dir):
        shutil.rmtree(args.original_data_dir)
    if os.path.exists(args.perturbed_data_dir):
        shutil.rmtree(args.perturbed_data_dir)
    os.makedirs(args.original_data_dir, exist_ok=True)
    os.makedirs(args.perturbed_data_dir, exist_ok=True)
    
    # load original dataset
    train_loader, test_loader = load_dataset(args.dataset, batch_size, shuffle=False)
    example_iter = iter(train_loader)
    first_batch, _ = next(example_iter)
    _, _, img_height, img_width = first_batch.size()
    
    # load trained model
    model = make_model(args.model, args.dataset).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    all_coverage_before = []
    all_coverage_after = []

    for batch_idx, (data, label) in enumerate(train_loader):
        # data = data.to(device)
        # print("##")
        # print(data.size)
        # tensor2png(data[1])
        # print(model.get_coverage(data, 1))
        # exit()

        # load data
        data = data.to(device)
        label = label.to(device)

        # initialzie noise vector
        noise = torch.randn((data[0].size()[0], 1, img_height, img_width), requires_grad=True, device=device)
        optimizer_noise = optim.Adam([noise], lr=1e-2)

        # save original images
        original_data_path = os.path.join(args.original_data_dir, f'original_data_batch_{batch_idx}_image.jpg')
        for i in range(data.size(0)):
            img = data[i].cpu().detach().numpy().transpose((1, 2, 0))
            img_path = original_data_path.replace(".jpg", f"_{i}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor((img + 1) * 127.5, cv2.COLOR_RGB2BGR))
        
        # Compute coverage (original images)
        coverage_before = model.get_coverage(data, 1)
        all_coverage_before.append(coverage_before)
        

        # Optimize noise iteratively
        for _ in range(args.step_num):
            # init optimizer
            optimizer_noise.zero_grad()
            
            # perturbed image
            perturbed_data = data + noise

            # apply constraints
            perturbed_data = torch.clamp(perturbed_data, -1, 1)

            # compute coverage (perturbed images)
            coverage = model.get_coverage(perturbed_data, 1)

            # our loss function
            loss_noise = torch.norm(noise) * args.lagrangian - coverage.mean()

            # optimize
            loss_noise.backward()
            optimizer_noise.step()

        # save perturbed images
        perturbed_data_path = os.path.join(args.perturbed_data_dir, f'perturbed_data_batch_{batch_idx}_image.jpg')
        for i in range(perturbed_data.size(0)):
            img = perturbed_data[i].cpu().detach().numpy().transpose((1, 2, 0))
            img_path = perturbed_data_path.replace(".jpg", f"_{i}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor((img + 1) * 127.5, cv2.COLOR_RGB2BGR))

        coverage_after = model.get_coverage(perturbed_data, 1)
        all_coverage_after.append(coverage_after)

    all_coverage_before_final = torch.cat(all_coverage_before)
    all_coverage_after_final = torch.cat(all_coverage_after)
    print("Coverage Before attack: {:.2f}%".format(all_coverage_before_final.mean() * 100))
    print("Coverage After attack: {:.2f}%".format(all_coverage_after_final.mean() * 100))

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
    parser.add_argument("-s", "--step-num", default=100, type=int)
    parser.add_argument("-l", "--lagrangian", default=0.01, type=float)
    parser.add_argument("--original-data-dir", default="original_train_data")
    parser.add_argument("--perturbed-data-dir", default="perturbed_train_data")
    parser.add_argument("-r", "--retraining-epoch", default=10, type=int)

    args = parser.parse_args()

    main(args)
