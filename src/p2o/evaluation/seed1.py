import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(3)
        self.fc2 = nn.Linear(3, 1, bias=False)
        
    def forward(self, input):
        x = self.fc1(input)
        x = self.bn1(x)  # Batch normalization
        x = torch.relu(x)  # Using ReLU activation function
        x = self.fc2(x)
        output = torch.sigmoid(x) * 5  # Scale output to range [0, 10]
        return output