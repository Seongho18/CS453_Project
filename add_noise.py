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

    pretrained_model = make_model(args.model, args.dataset).to(device)
    pretrained_model.load_state_dict(torch.load(args.model_path))
    pretrained_model.eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(pretrained_model.parameters(), lr=args.learning_rate)


    all_coverage_before = []
    all_coverage_after = []

    perturbed_images_list = []

    for epoch in range(args.epochs_noise):
        if epoch == 0:
            for batch_idx, (data, label) in enumerate(train_loader):
                # load data
                data = data.to(device)
                label = label.to(device)

                # initialize noise vector
                noise = torch.randn((data.size()[0], 1, img_height, img_width), requires_grad=True, device=device)
                optimizer_noise = optim.Adam([noise], lr=1e-2)

                # save original images
                if batch_idx == 0:
                    original_data_path = os.path.join(args.original_data_dir, f'original_data_batch_{batch_idx}_image.jpg')
                    img = data[0].cpu().detach().numpy().transpose((1, 2, 0))
                    img_path = original_data_path.replace(".jpg", f".jpg")
                    cv2.imwrite(img_path, cv2.cvtColor((img + 1) * 127.5, cv2.COLOR_RGB2BGR))
                
                # Compute coverage (original images)
                coverage_before = model.get_coverage(data, 0.7)
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
                    coverage = model.get_coverage(perturbed_data, 0.7)

                    # our loss function
                    loss_noise = torch.norm(noise) * args.lagrangian - coverage.mean()

                    # optimize
                    loss_noise.backward()
                    optimizer_noise.step()

                # store perturbed images in memory
                perturbed_images_list.append((perturbed_data.clone(), label.clone()))

                coverage_after = model.get_coverage(perturbed_data, 0.7)
                all_coverage_after.append(coverage_after)

                optimizer.zero_grad()
                outputs, _ = pretrained_model(perturbed_data.detach())
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()
                
            all_coverage_before_final = torch.cat(all_coverage_before)
            all_coverage_after_final = torch.cat(all_coverage_after)
            print("Coverage Before attack: {:.2f}%".format(all_coverage_before_final.mean() * 100))
            print("Coverage After attack: {:.2f}%".format(all_coverage_after_final.mean() * 100))
            
        else:
            for perturbed_data, label in perturbed_images_list:
                perturbed_data = perturbed_data.to(device)
                label = label.to(device)
                
                optimizer.zero_grad()
                outputs, _ = pretrained_model(perturbed_data.detach())
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

        accuracy = calculate_accuracy(pretrained_model, test_loader, device)
        print(f'Epoch [{epoch+1}/{args.epochs_noise}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

    # save model
    save_path = os.path.join(args.save_path, f"{args.model}_{args.dataset}_with_noise_{args.epochs_noise}epoch.pt")
    torch.save(model.state_dict(), save_path)
    end = time.time()
    print(f"{end - start:.5f} sec")

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
    parser.add_argument("--perturbed-data-dir", default="perturbed_train_data")
    parser.add_argument("-e", "--epochs-noise", type=int, default=10)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.001)

    args = parser.parse_args()

    main(args)
