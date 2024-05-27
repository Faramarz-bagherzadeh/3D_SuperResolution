import torch
import torch.nn as nn
#from torchsummary import summary

class DenseBlock3D(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            # Calculate the input channels for each layer in the block
            input_channels = in_channels + i * growth_rate

            # Define a densely connected layer with batch normalization and ELU activation
            layer = nn.Sequential(
                nn.BatchNorm3d(input_channels),
                nn.ELU(),
                nn.Conv3d(input_channels, growth_rate, kernel_size=3,padding='same')
            )
            self.layers.append(layer)

    def forward(self, x):
        # Store the input for skip connections
        skip_connections = [x]

        # Forward pass through each layer in the block
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
            skip_connections.append(x)

        # Return the concatenated feature maps and skip connections
        return torch.cat(skip_connections, dim=1)



class DCSRN(nn.Module):
    def __init__(self, in_channels):
        super(DCSRN, self).__init__()
        growth_rate = 8
        num_layers = 4
        num_filters = 48

        # 3D Convolutional Layer
        self.conv3d_layer_1 = nn.Conv3d (in_channels, num_filters, kernel_size=3, padding='same')

        # 3D Dense Block
        self.dense_block_3d = DenseBlock3D(num_filters, growth_rate, num_layers)
        
        
        self.conv3d_layer_2 = nn.Conv3d (320, 1, kernel_size=1, padding='same')

    def forward(self, x):
        x = self.conv3d_layer_1(x)

        x = self.dense_block_3d(x)
        
        x = self.conv3d_layer_2(x)

        return x







# Assuming your model is currently on the GPU
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Creating an instance of the model
#model = DCSRN(1)
# Move the model to the CPU
#model.to('cpu')
#     Print the model summary

#summary(model, input_size=(1, 16, 16, 16), device="cpu")



