import argparse
import os

import torch
import torch.optim as optim

from data.utils import *
from model.utils import *


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    learning_rate = args.learning_rate
    epochs = args.epochs
    batch_size = args.batch

    # load dataset
    print(f"Training on {args.dataset}")
    train_loader, test_loader = load_dataset(args.dataset, batch_size)

    # construct model, optimizer
    model = make_model(args.model, args.dataset).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # train model
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # save model
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

    args = parser.parse_args()

    main(args)
