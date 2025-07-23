import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """基础卷积块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class EncoderModule(nn.Module):
    """编码器模块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(EncoderModule, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = ConvBlock(out_channels, out_channels)
        
        # 残差连接
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
            
        out += residual
        return F.relu(out)


class EncoderBackbone(nn.Module):
    """编码器主干网络，整体步幅16"""
    def __init__(self, in_channels=3, base_channels=64):
        super(EncoderBackbone, self).__init__()
        
        # 初始卷积
        self.initial_conv = ConvBlock(in_channels, base_channels, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 四个编码器模块
        # Module 1: stride=1 (相对于maxpool后)
        self.module1 = EncoderModule(base_channels, base_channels, stride=1)
        
        # Module 2: stride=2 (累计stride=4)
        self.module2 = EncoderModule(base_channels, base_channels * 2, stride=2)
        
        # Module 3: stride=2 (累计stride=8)  
        self.module3 = EncoderModule(base_channels * 2, base_channels * 4, stride=2)
        
        # Module 4: stride=1 (保持stride=16)
        self.module4 = EncoderModule(base_channels * 4, base_channels * 8, stride=1)
        
        self.base_channels = base_channels
        
    def forward(self, x):
        # 初始处理
        x = self.initial_conv(x)  # stride 2
        x = self.maxpool(x)       # stride 4
        
        # 四个模块
        b1 = self.module1(x)      # stride 4
        b2 = self.module2(b1)     # stride 8
        b3 = self.module3(b2)     # stride 16
        b4 = self.module4(b3)     # stride 16
        
        return {
            'b1': b1,  # [B, base_channels, H/4, W/4]
            'b2': b2,  # [B, base_channels*2, H/8, W/8]
            'b3': b3,  # [B, base_channels*4, H/16, W/16]
            'b4': b4   # [B, base_channels*8, H/16, W/16]
        }
    
    def get_feature_channels(self):
        """返回各层特征通道数"""
        return {
            'b1': self.base_channels,
            'b2': self.base_channels * 2,
            'b3': self.base_channels * 4,
            'b4': self.base_channels * 8
        }
