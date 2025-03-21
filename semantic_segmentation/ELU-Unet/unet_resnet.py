import torch
import numpy as np
from torch import nn
import random
import torchvision.models as models

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

class Down(nn.Module):
    # Contracting Layer
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    # Expanding Layer
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class PretrainedResNetBottleneck(nn.Module):
    """ ResNet-based Bottleneck using a Pretrained Model """
    def __init__(self, in_channels):
        super(PretrainedResNetBottleneck, self).__init__()
        
        # Load Pretrained ResNet
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Use ResNet-18
        self.resnet_layers = nn.Sequential(*list(resnet.children())[4:6])  
        
        self.input_conv = nn.Conv2d(in_channels, 64, kernel_size=1)
        
        self.output_conv = nn.Conv2d(128, in_channels, kernel_size=1)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True) 

    def forward(self, x):
        x = self.input_conv(x)
        x = self.resnet_layers(x)
        x = self.output_conv(x)
        x = self.upsample(x)
        return x

class UNet_Resnet(nn.Module):
    def __init__(self, input_bands=11, output_classes=11, hidden_channels=16):
        super(UNet_Resnet, self).__init__()

        # Initial Convolution Layer
        self.inc = nn.Sequential(
            nn.Conv2d(input_bands, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True))

        # Contracting Path
        self.down1 = Down(hidden_channels, 2 * hidden_channels)
        self.down2 = Down(2 * hidden_channels, 4 * hidden_channels)
        self.down3 = Down(4 * hidden_channels, 8 * hidden_channels)
        self.down4 = Down(8 * hidden_channels, 8 * hidden_channels)

        # ResNet Bottleneck
        self.resnet_bottleneck = PretrainedResNetBottleneck(8 * hidden_channels)

        # Expanding Path
        self.up1 = Up(16 * hidden_channels, 4 * hidden_channels)
        self.up2 = Up(8 * hidden_channels, 2 * hidden_channels)
        self.up3 = Up(4 * hidden_channels, hidden_channels)
        self.up4 = Up(2 * hidden_channels, hidden_channels)

        # Output Convolution Layer
        self.outc = nn.Conv2d(hidden_channels, output_classes, kernel_size=1)

    def forward(self, x):
        # Initial Convolution Layer
        x1 = self.inc(x)

        # Contracting Path
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.resnet_bottleneck(x5)
        # Expanding Path
        x7 = self.up1(x6, x4)
        x8 = self.up2(x7, x3)
        x9 = self.up3(x8, x2)
        x10 = self.up4(x9, x1)

        # Output Convolution Layer
        return self.outc(x10)
