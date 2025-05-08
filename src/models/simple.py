# do a basic convolutional neural network
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size, kernel_size):
        super(SimpleCNN, self).__init__()
        layers = []
        # stride = 1 by default
        # padding = kernel_size//2 to keep the same size and preserve temporal length
        layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size=kernel_size, padding=kernel_size//2))
        # using ReLU activation as a module to prevent Captum crash
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool1d(2))
        layers.append(nn.Conv1d(hidden_size, out_channels, kernel_size=kernel_size, padding=0))
        layers.append(nn.AdaptiveAvgPool1d(1))
        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        
    def forward(self, x):
        # making sure at least it is batch_size, channels, time
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
        return self.cnn(x)
    


