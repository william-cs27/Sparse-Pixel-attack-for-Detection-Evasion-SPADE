import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config

class REINFORCE(nn.Module):
    def __init__(self, config):
        super(REINFORCE, self).__init__()
        self.data = []
        self.r = []
        self.prob = []

        # Convolutional layers
        self.conv1 = nn.Conv2d(config["RGB"], 32, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.relu = nn.ReLU()

        # Find out dynamically the size for the FC layer
        # Assume input will be shape [B, 3, H, W]
        # Use your transform output size 
        dummy_input = torch.zeros(1, config["RGB"], 224, 224)  
        with torch.no_grad():
            x = self.conv1(dummy_input)
            x = self.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.conv2(x)
            x = self.relu(x)
            x = F.max_pool2d(x, 2)
            n_flatten = x.view(1, -1).shape[1]

        # Fully connected layer (use calculated n_flatten)
        self.fc = nn.Linear(n_flatten, 512)

        # Action mean and log std layers
        self.action_mean = torch.nn.Linear(512, 5)
        self.action_logstd = torch.nn.Linear(512, 5)

        self.device = config["device"]
        self.optimizer = torch.optim.SGD(self.parameters(), lr=config["RL_learning_rate"])
    def forward(self, state):
        image = state.clone().detach().to(self.device)
        image = self.conv1(image)
        image = self.relu(image)
        image = F.max_pool2d(image, 2)
        image = self.conv2(image)
        image = self.relu(image)
        image = F.max_pool2d(image, 2)
        image = torch.flatten(image, 1)
        image = self.fc(image)
        image = self.relu(image)
        action_mean = self.action_mean(image)
        action_logstd = self.action_logstd(image)
        # Compute action std
        if config["classifier"] == "yolo" or config["classifier"] == "ddq":
            action_std = 2 * torch.sigmoid(action_logstd)
        else:
            action_std = torch.exp(action_logstd)
        return action_mean, action_std

    def train_net(self):
        self.optimizer.zero_grad()
        loss = -1 * self.prob.to(self.device) * self.r
        loss.sum().backward()
        self.optimizer.step()
        self.r, self.prob = [], []
