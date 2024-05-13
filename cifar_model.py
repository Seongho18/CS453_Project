import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR_Model(nn.Module):

    def __init__(self, n_classes):
        super(CIFAR_Model, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3),
            nn.ReLU()
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3),
            nn.ReLU()
        )
        self.pooling1 = nn.MaxPool2d(kernel_size=2)
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.ReLU()
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3),
            nn.ReLU()
        )
        self.pooling2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.linear1 = nn.Sequential(
            nn.Linear(in_features=3200, out_features=256),
            nn.ReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU()
        )
        self.linear3 = nn.Linear(in_features=256, out_features=n_classes),

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.pooling1(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.pooling2(x)
        x = self.linear1(x)
        x = self.linear2(x)
        logits = self.linear3(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs

    def _count_neurons(self, layer, x, threshold, num_neurons, num_activated):
        x = layer(x)
        num_neurons = torch.numel(x)
        num_activated += torch.sum(x > threshold)
        return x, num_neurons, num_activated

    def get_coverage(self, x, threshold):
        num_neurons = 0
        num_activated = 0
        x, num_neurons, num_activated = self._count_neurons(
            self.cnn1, x, threshold, num_neurons, num_activated
        )
        x, num_neurons, num_activated = self._count_neurons(
            self.cnn2, x, threshold, num_neurons, num_activated
        )
        x = self.pooling1(x)
        x, num_neurons, num_activated = self._count_neurons(
            self.cnn3, x, threshold, num_neurons, num_activated
        )
        x, num_neurons, num_activated = self._count_neurons(
            self.cnn4, x, threshold, num_neurons, num_activated
        )
        x = self.pooling2(x)
        x, num_neurons, num_activated = self._count_neurons(
            self.linear1, x, threshold, num_neurons, num_activated
        )
        x, num_neurons, num_activated = self._count_neurons(
            self.linear2, x, threshold, num_neurons, num_activated
        )
        x, num_neurons, num_activated = self._count_neurons(
            self.linear3, x, threshold, num_neurons, num_activated
        )
        coverage = num_activated / num_neurons
        return coverage
