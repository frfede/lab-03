import torch
from torch import nn

# Define the custom neural network
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # Define layers of the neural network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=3)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=0, stride=3)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=3)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
        # Define forward pass

        # B x 3 x 224 x 224
        x = self.conv1(x).relu() # B x 64 x 75 x 75
        x = self.conv2(x).relu() # B x 128 x 25 x 25
        x = self.conv3(x).relu() # B x 256 x 9 x 9

        x = self.pool(x)

        x = torch.flatten(x, 1) # B x 256
        x = self.fc1(x) # B x 200

        return x