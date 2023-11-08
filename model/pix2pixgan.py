import torch
import torch.nn as nn
import torch.nn.functional as F 
#from torch.utils.tensorboard import SummaryWriter



class DoubleConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBNReLU, self).__init__()
        self.doubleconv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),            
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        x = self.doubleconv_bn_relu(x)
        return x


class Down(nn.Module):
    def __init__(self):
        super(Down, self).__init__()
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.down(x)
        return x

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        x = self.up(x)
        x = self.conv_bn_relu(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.conv_relu = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        x = self.conv_relu(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.doubleconv1 = DoubleConvBNReLU(1, 64)
        self.down1 = Down()
        self.doubleconv2 = DoubleConvBNReLU(64, 128)
        self.down2 = Down()
        self.doubleconv3 = DoubleConvBNReLU(128, 256)
        self.down3 = Down()
        self.doubleconv4 = DoubleConvBNReLU(256, 512)
        self.down4 = Down()
        self.doubleconv5 = DoubleConvBNReLU(512, 1024)
        self.upconv1 = UpConv(1024, 512)
        self.doubleconv6 = DoubleConvBNReLU(1024, 512)
        self.upconv2 = UpConv(512, 256)
        self.doubleconv7 = DoubleConvBNReLU(512, 256)
        self.upconv3 = UpConv(256, 128)
        self.doubleconv8 = DoubleConvBNReLU(256, 128)
        self.upconv4 = UpConv(128, 64)
        self.doubleconv9 = DoubleConvBNReLU(128, 64)
        self.last = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid())


    def forward(self, x):
        x1 = self.doubleconv1(x)
        x2 = self.down1(x1)
        x2 = self.doubleconv2(x2)
        x3 = self.down2(x2)
        x3 = self.doubleconv3(x3)
        x4 = self.down3(x3)
        x4 = self.doubleconv4(x4)
        x5 = self.down4(x4)
        x5 = self.doubleconv5(x5)

        x6 = self.upconv1(x5)
        x6 = torch.cat([x4, x6], dim=1)
        x6 = self.doubleconv6(x6)

        x7 = self.upconv2(x6)
        x7 = torch.cat([x3, x7], dim=1)
        x7 = self.doubleconv7(x7)

        x8 = self.upconv3(x7)
        x8 = torch.cat([x2, x8], dim=1)
        x8 = self.doubleconv8(x8)

        x9 = self.upconv4(x8)
        x9 = torch.cat([x1, x9], dim=1)
        x9 = self.doubleconv9(x9)
        
        x10 = self.last(x9)

        return x10


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.down1 = Downsample(2, 64)
        self.donw2 = Downsample(64, 128)
        self.down3 = Downsample(128, 256)
        self.conv = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(512)
        self.last = nn.Conv2d(512, 1, kernel_size=3, stride=1)

    def forward(self, label, img):
        x = torch.cat([label, img], dim=1)
        x = self.down1(x)
        x = self.donw2(x)
        #x = F.dropout2d(self.down3(x))
        #x = F.dropout2d(F.Leaky_relu(self.conv(x)))
        #x = F.dropout2d(self.bn(x))
        x = self.down3(x)
        x = self.conv(x)
        x = self.bn(x)
        x = torch.sigmoid(self.last(x))
        return x
