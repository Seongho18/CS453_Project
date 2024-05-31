import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = [batch size, height, width]
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        # h_1 = [batch size, 250]
        h_1 = F.relu(self.input_fc(x))

        # h_2 = [batch size, 100]
        h_2 = F.relu(self.hidden_fc(h_1))

        # y_pred = [batch size, output dim]
        y_pred = self.output_fc(h_2)

        return y_pred, h_2

    def get_coverage(self, x, threshold):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        
        h_1 = F.relu(self.input_fc(x))
        activated_1 = self.sigmoid(10*(h_1 - threshold))

        h_2 = F.relu(self.hidden_fc(h_1))
        activated_2 = self.sigmoid(10*(h_2 - threshold))

        # [batch size, 350]
        activated = torch.cat([activated_1, activated_2], 1)
        # [batch size]
        coverage = torch.mean(activated, 1)
        return coverage

class LeNet_MNIST(nn.Module):
    def __init__(self, input_channels=1, output_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc_1 = nn.Linear(16 * 4 * 4, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.shape[0], -1)
        h = x
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        
        return x, h
    
    def get_coverage(self, x, threshold):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        activated_1 = torch.flatten(self.sigmoid(10*(x - threshold)), start_dim=1)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        activated_2 = torch.flatten(self.sigmoid(10*(x - threshold)), start_dim=1)
        x = x.view(x.shape[0], -1)
        h = x
        x = F.relu(self.fc_1(x))
        activated_3 = self.sigmoid(10*(x - threshold))
        x = F.relu(self.fc_2(x))
        activated_4 = self.sigmoid(10*(x - threshold))
        x = self.fc_3(x)

        activated = torch.cat([activated_1, activated_2, activated_3, activated_4], 1)
        coverage = torch.mean(activated, 1)
        return coverage

class LeNet_CIFAR10(nn.Module):
    def __init__(self, input_channels=3, output_dim=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        
        # To calculate the input size of the first fully connected layer
        self._to_linear = None
        self.convs = nn.Sequential(
            self.conv1,
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            self.conv2,
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        x = torch.randn(input_channels, 32, 32).unsqueeze(0)
        self._to_linear = self.convs(x).view(-1).shape[0]

        self.fc_1 = nn.Linear(self._to_linear, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        h = x
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = self.fc_3(x)
        return x, h

    def get_coverage(self, x, threshold):
        x = self.conv1(x)
        activated_1 = torch.flatten(self.sigmoid(10*(x - threshold)), start_dim=1)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))

        x = self.conv2(x)
        activated_2 = torch.flatten(self.sigmoid(10*(x - threshold)), start_dim=1)
        x = F.relu(x)
        x = F.max_pool2d(x, (2,2))
        x = x.view(x.shape[0], -1)


        x = self.fc_1(x)
        activated_3 = self.sigmoid(10*(x - threshold))
        x = F.relu(x)
        
        x = self.fc_2(x)
        activated_4 = self.sigmoid(10*(x - threshold))
        x = F.relu(x)
        x = self.fc_3(x)

        activated = torch.cat([activated_1, activated_2, activated_3, activated_4], 1)
        coverage = torch.mean(activated, 1)
        return coverage
