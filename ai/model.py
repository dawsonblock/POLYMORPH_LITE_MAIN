import torch
import torch.nn as nn

class RamanCNN(nn.Module):
    """
    1D Convolutional Neural Network for Raman Spectrum Analysis.
    Predicts concentration (0.0 - 1.0) from spectral data.
    """
    def __init__(self, input_size=1024):
        super(RamanCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        
        # Calculate size after pooling: 1024 -> 512 -> 256
        self.fc1 = nn.Linear(32 * 256, 64)
        self.fc2 = nn.Linear(64, 1) # Regression output (concentration)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, input_size) -> (batch_size, 1, input_size)
        x = x.unsqueeze(1)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x
