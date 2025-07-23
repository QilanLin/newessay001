import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderS(nn.Module):
    """小型解码器 S (×2上采样)"""
    def __init__(self, in_channels, out_channels=1):
        super(DecoderS, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.conv1x1 = nn.Conv2d(in_channels // 2, out_channels, 1)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        return x


class DecoderM(nn.Module):
    """中型解码器 M (×4上采样)"""
    def __init__(self, in_channels, out_channels=1):
        super(DecoderM, self).__init__()
        
        # 第一次上采样×2
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 第二次上采样×2
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        self.conv1x1 = nn.Conv2d(in_channels // 4, out_channels, 1)
        
    def forward(self, x):
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.conv1x1(x)
        return x


class DecoderL3(nn.Module):
    """大型解码器 L3 (×8上采样)"""
    def __init__(self, in_channels, out_channels=1):
        super(DecoderL3, self).__init__()
        
        # 第一次上采样×2
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 第二次上采样×2
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 第三次上采样×2
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 8, 3, padding=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        
        self.conv1x1 = nn.Conv2d(in_channels // 8, out_channels, 1)
        
    def forward(self, x):
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.upsample3(x)
        x = self.conv3(x)
        x = self.conv1x1(x)
        return x


class DecoderL4(nn.Module):
    """大型解码器 L4 (×16上采样)"""
    def __init__(self, in_channels, out_channels=1):
        super(DecoderL4, self).__init__()
        
        # 第一次上采样×2
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 第二次上采样×2
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True)
        )
        
        # 第三次上采样×2
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels // 8, 3, padding=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(inplace=True)
        )
        
        # 第四次上采样×2
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels // 8, in_channels // 16, 3, padding=1),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(inplace=True)
        )
        
        self.conv1x1 = nn.Conv2d(in_channels // 16, out_channels, 1)
        
    def forward(self, x):
        x = self.upsample1(x)
        x = self.conv1(x)
        x = self.upsample2(x)
        x = self.conv2(x)
        x = self.upsample3(x)
        x = self.conv3(x)
        x = self.upsample4(x)
        x = self.conv4(x)
        x = self.conv1x1(x)
        return x


class MaskFusion(nn.Module):
    """掩码融合与预测模块"""
    def __init__(self, num_decoders=4):
        super(MaskFusion, self).__init__()
        
        # 四路串接后的1×1卷积
        self.fusion_conv = nn.Conv2d(num_decoders, 1, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, decoder_outputs, target_size):
        """
        Args:
            decoder_outputs: 包含四个解码器输出的列表
        """
        # 确保所有输出都是相同尺寸（1×）
        aligned_outputs = []
        
        for output in decoder_outputs:
            if output.shape[2:] != target_size:
                output = F.interpolate(
                    output, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            aligned_outputs.append(output)
        
        # 四路串接
        concatenated = torch.cat(aligned_outputs, dim=1)
        
        # 1×1卷积 + Sigmoid
        fused = self.fusion_conv(concatenated)
        mask_prediction = self.sigmoid(fused)
        
        return mask_prediction


class MultiResolutionDecoders(nn.Module):
    """多分辨率解码器集合"""
    def __init__(self, msc_channels=256):
        super(MultiResolutionDecoders, self).__init__()
        
        # 四个解码器
        self.decoder_s = DecoderS(msc_channels, out_channels=1)      # ×2
        self.decoder_m = DecoderM(msc_channels, out_channels=1)      # ×4
        self.decoder_l3 = DecoderL3(msc_channels, out_channels=1)    # ×8
        self.decoder_l4 = DecoderL4(msc_channels, out_channels=1)    # ×16
        
        # 掩码融合
        self.mask_fusion = MaskFusion(num_decoders=4)
        
    def forward(self, msc_features, input_size):
        """
        Args:
            msc_features: MSCFE输出的特征字典
            input_size: 输入图像的(H, W)尺寸，用于将所有解码器输出对齐到1×
        """
        # 各解码器处理
        ds_out = self.decoder_s(msc_features['msc1'])    # 来自MSC1
        dm_out = self.decoder_m(msc_features['msc2'])    # 来自MSC2
        dl3_out = self.decoder_l3(msc_features['msc3'])  # 来自MSC3
        dl4_out = self.decoder_l4(msc_features['msc4'])  # 来自MSC4
        
        # 掩码融合到输入尺寸
        mask_prediction = self.mask_fusion([
            ds_out,
            dm_out,
            dl3_out,
            dl4_out
        ], target_size=input_size)
        
        return {
            'ds_out': ds_out,
            'dm_out': dm_out,
            'dl3_out': dl3_out,
            'dl4_out': dl4_out,
            'mask': mask_prediction
        }
