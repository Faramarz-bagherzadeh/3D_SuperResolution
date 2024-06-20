import torch
import torch.nn as nn


# Define the basic residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding='same', bias=False)
       # self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False)
       # self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Shortcut connection to handle different dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
               # nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
      #  out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
       # out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

# Define the SRResNet architecture
class SRResNet(nn.Module):
    def __init__(self, num_blocks=16, upscale_factor=2):
        super(SRResNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64, 64) for _ in range(num_blocks)])
        
        # Additional convolution layers
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn = nn.BatchNorm3d(64)
        
        # Upscaling layer
        #self.upscale = nn.Sequential(
        #    nn.Conv3d(64, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #    nn.PixelShuffle(upscale_factor),
        #    nn.ReLU(inplace=True))
        
        # Output layer
        self.conv3 = nn.Conv3d(64, 1, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        residual = out
        
        out = self.residual_blocks(out)
        
        out = self.conv2(out)
        #out = self.bn(out)
        out += residual
        
        #out = self.upscale(out)
        out = self.conv3(out)
        zoom = int(out.shape[-1]/4)
        out = out[:,:,zoom:-zoom,zoom:-zoom,zoom:-zoom]        
        return out





from torchsummary import summary
# Assuming your model is currently on the GPU
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Creating an instance of the model
model = SRResNet()
#model = ResidualBlock(1, 1)
# Move the model to the CPU
model.to('cpu')
#     Print the model summary

summary(model, input_size=(1, 16, 16, 16), device="cpu")































