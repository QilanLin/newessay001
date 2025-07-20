import torch
import torch.nn as nn
import torch.nn.functional as F


class CFPNLevel(nn.Module):
    """CFPN单层处理"""
    def __init__(self, in_channels, out_channels):
        super(CFPNLevel, self).__init__()
        
        # 高分辨率路径 (stride 1)
        self.high_res_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 低分辨率路径 (stride 2)
        self.low_res_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 高分辨率路径
        high_res = self.high_res_conv(x)
        
        # 低分辨率路径
        low_res = self.low_res_conv(x)
        
        # 尺寸对齐：将低分辨率上采样到高分辨率尺寸
        low_res_aligned = F.interpolate(
            low_res, 
            size=high_res.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        # 串接
        concatenated = torch.cat([high_res, low_res_aligned], dim=1)
        
        # 1x1卷积融合
        fused = self.fusion_conv(concatenated)
        
        return fused


class CFPN(nn.Module):
    """对比特征金字塔网络"""
    def __init__(self, feature_channels, out_channels=256):
        super(CFPN, self).__init__()
        
        # 为B2, B3, B4创建CFPN层
        self.cfpn_level2 = CFPNLevel(feature_channels['b2'], out_channels)
        self.cfpn_level3 = CFPNLevel(feature_channels['b3'], out_channels)
        self.cfpn_level4 = CFPNLevel(feature_channels['b4'], out_channels)
        
        self.out_channels = out_channels
        
    def forward(self, features):
        """
        Args:
            features: dict with keys 'b1', 'b2', 'b3', 'b4'
        Returns:
            dict with fused features for levels 2, 3, 4
        """
        fuse2 = self.cfpn_level2(features['b2'])
        fuse3 = self.cfpn_level3(features['b3'])
        fuse4 = self.cfpn_level4(features['b4'])
        
        return {
            'fuse2': fuse2,  # 来自B2的融合特征
            'fuse3': fuse3,  # 来自B3的融合特征
            'fuse4': fuse4   # 来自B4的融合特征
        }
