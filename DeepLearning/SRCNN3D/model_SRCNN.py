import torch
import torch.nn as nn

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv3d(1, 64, kernel_size=9, padding='same')
        self.conv2 = nn.Conv3d(64, 32, kernel_size=1, padding='same')
        self.conv3 = nn.Conv3d(32, 1, kernel_size=5, padding='same')

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)

        return x

# Instantiate the SRCNN model
#model = SRCNN()

# Print the model architecture
#print(model)