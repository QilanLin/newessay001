import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDecoupling(nn.Module):
    """特征解耦模块 - 从B4生成前景、背景和不确定性原型"""
    def __init__(self, in_channels, prototype_dim=256):
        super(FeatureDecoupling, self).__init__()
        
        # 前景分支
        self.fg_branch = nn.Sequential(
            nn.Conv2d(in_channels, prototype_dim, 3, padding=1),
            nn.BatchNorm2d(prototype_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(prototype_dim, prototype_dim, 1),
            nn.BatchNorm2d(prototype_dim),
            nn.ReLU(inplace=True)
        )
        
        # 背景分支
        self.bg_branch = nn.Sequential(
            nn.Conv2d(in_channels, prototype_dim, 3, padding=1),
            nn.BatchNorm2d(prototype_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(prototype_dim, prototype_dim, 1),
            nn.BatchNorm2d(prototype_dim),
            nn.ReLU(inplace=True)
        )
        
        # 不确定性分支
        self.uc_branch = nn.Sequential(
            nn.Conv2d(in_channels, prototype_dim, 3, padding=1),
            nn.BatchNorm2d(prototype_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(prototype_dim, prototype_dim, 1),
            nn.BatchNorm2d(prototype_dim),
            nn.ReLU(inplace=True)
        )
        
        self.prototype_dim = prototype_dim
        
    def forward(self, b4_features):
        """
        Args:
            b4_features: 编码器第4层特征 [B, C, H/16, W/16]
        Returns:
            dict: 包含三个原型特征的字典
        """
        # 生成三个原型
        f_fg = self.fg_branch(b4_features)    # 前景原型
        f_bg = self.bg_branch(b4_features)    # 背景原型
        f_uc = self.uc_branch(b4_features)    # 不确定性原型
        
        return {
            'f_fg': f_fg,
            'f_bg': f_bg,
            'f_uc': f_uc
        }
    
    def get_prototype_channels(self):
        """返回原型特征通道数"""
        return self.prototype_dim
