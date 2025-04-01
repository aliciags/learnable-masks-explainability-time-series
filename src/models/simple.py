# do a basic convolutional neural network
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(1500, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 1500)
        x = self.fc1(x)
        return x
    


