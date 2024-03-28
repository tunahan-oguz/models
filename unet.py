import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_chan=None) -> None:
        super().__init__()
        mid_chan = out_channels if mid_chan is None else mid_chan
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_chan, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_chan, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, in_channels // 2)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
    


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, mid_chan=None) -> None:
        super().__init__()
        mid_chan = out_channels if mid_chan is None else mid_chan
        self.MaxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv = DoubleConv(in_channels, out_channels, mid_chan)
    
    def forward(self, x):
        return self.conv(self.MaxPool(x))


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.input_layer = DoubleConv(in_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024)
        self.up2 = Up(512)
        self.up3 = Up(256)
        self.up4 = Up(128)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        x = self.input_layer(x)
        
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        upout = self.up1(d4, d3)
        upout = self.up2(upout, d2)
        upout = self.up3(upout, d1)
        upout = self.up4(upout, x)

        return self.final_conv(upout)
