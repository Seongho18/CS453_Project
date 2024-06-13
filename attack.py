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

import time


def x2w(x):
    '''
    Transform pixel value into continuous space -inf ~ inf
    The algorithm is originated from [Nicholas Carlini, David Wagner 
    (2017) Towards Evaluating the Robustness of Neural Networks]
    '''
    # x : [-1, 1],   w : [-inf, inf]
    w = torch.tanh(x.detach())
    w = w.clone()
    w.requires_grad_()
    return w

def w2x(w):
    x = torch.atanh(w)
    return x


def calculate_accuracy(model, data_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs, _ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 64

    # Create directories for saving images
    if os.path.exists(args.original_data_dir):
        shutil.rmtree(args.original_data_dir)
    if os.path.exists(args.adversarial_data_dir):
        shutil.rmtree(args.adversarial_data_dir)
    os.makedirs(args.original_data_dir, exist_ok=True)
    os.makedirs(args.adversarial_data_dir, exist_ok=True)
    
    # load original dataset
    train_loader, test_loader = load_dataset(args.dataset, batch_size, shuffle=False)
    example_iter = iter(train_loader)
    first_batch, _ = next(example_iter)
    _, _, img_height, img_width = first_batch.size()
    
    # load trained model
    model = make_model(args.model, args.dataset).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()


    success_num = 0
    total_num = 0
    perturbed_images_list = []


    for batch_idx, (data, label) in enumerate(train_loader):
        total_num += data.shape[0]
        # load data
        data = data.to(device)
        label = label.to(device)

        # initialize adverarial image
        w = x2w(data)
        optimizer = optim.Adam([w], lr=1e-2)

        # print("ASDFASDF")
        # logit_vector, _ = model.forward(data)
        # perturbed_label = torch.argmax(logit_vector, dim=1)
        # print(perturbed_label)

        # save original images
        if batch_idx == 0:
            original_data_path = os.path.join(args.original_data_dir, f'original_data_batch_{batch_idx}_image.jpg')
            img = data[0].cpu().detach().numpy().transpose((1, 2, 0))
            img_path = original_data_path.replace(".jpg", f".jpg")
            cv2.imwrite(img_path, cv2.cvtColor((img + 1) * 127.5, cv2.COLOR_RGB2BGR))
        
        # Optimize noise iteratively
        for _ in range(100):
            # init optimizer
            optimizer.zero_grad()
            
            # perturbed image
            x = w2x(w)
            # compute objective function
            logit_vector, _ = model.forward(x)
            probs = torch.softmax(logit_vector, dim=1)
            prob = torch.stack([p[i] for p, i in zip(probs, label)])
            
            L2_dist = torch.norm(torch.flatten(x - data, start_dim=1), dim=1)
            # our loss function
            loss = L2_dist - 2*torch.log10(2 * prob - 2)
            loss = torch.sum(loss)
            # optimize
            loss.backward()
            optimizer.step()

        perturbed_data = w2x(w)
        # store perturbed images in memory
        perturbed_images_list.append((perturbed_data.clone(), label.clone()))


        # get attack success rate
        logit_vector, _ = model.forward(perturbed_data)
        perturbed_label = torch.argmax(logit_vector, dim=1)

        for y, y_ in zip(label, perturbed_label):
            if not y == y_:
                success_num += 1 
        # print(prob)
        # print(perturbed_label)
        # print(label)
        # exit()

        original_data_path = os.path.join(args.adversarial_data_dir , f'adversarial_data_batch_{batch_idx}_image.jpg')
        img = perturbed_data[0].cpu().detach().numpy().transpose((1, 2, 0))
        img_path = original_data_path.replace(".jpg", f".jpg")
        cv2.imwrite(img_path, cv2.cvtColor((img + 1) * 127.5, cv2.COLOR_RGB2BGR))


    print(success_num / total_num)


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["MNIST", "CIFAR10"], default="MNIST")
    parser.add_argument("-m", "--model", choices=["MLP", "LENET"], default="MLP")
    parser.add_argument(
        "--model-path",
        required=True,
        help="save path of model weight",
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help="save path of model weight",
        default="./weights/"
    )
    # args for noise
    parser.add_argument("-s", "--step-num", default=100, type=int)
    parser.add_argument("-l", "--lagrangian", default=0.01, type=float)
    parser.add_argument("--original-data-dir", default="original_train_data")
    parser.add_argument("--adversarial-data-dir", default="adversarial_data")
    parser.add_argument("-e", "--epochs-noise", type=int, default=10)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.001)

    args = parser.parse_args()

    main(args)
