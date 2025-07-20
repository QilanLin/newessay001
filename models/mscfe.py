import torch
import torch.nn as nn
import torch.nn.functional as F


class DilatedConvBlock(nn.Module):
    """空洞卷积块"""
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(DilatedConvBlock, self).__init__()
        
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=3, 
                padding=dilation_rate, 
                dilation=dilation_rate
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.dilated_conv(x)


class MSCBlock(nn.Module):
    """多尺度对比特征块"""
    def __init__(self, in_channels, out_channels):
        super(MSCBlock, self).__init__()
        
        # 多尺度空洞卷积
        self.dilated_conv_1 = DilatedConvBlock(in_channels, out_channels // 4, dilation_rate=1)
        self.dilated_conv_2 = DilatedConvBlock(in_channels, out_channels // 4, dilation_rate=2)
        self.dilated_conv_4 = DilatedConvBlock(in_channels, out_channels // 4, dilation_rate=4)
        self.dilated_conv_8 = DilatedConvBlock(in_channels, out_channels // 4, dilation_rate=8)
        
        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 原型反馈输入
        self.prototype_proj = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1),  # 3个原型特征
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # 损失反馈输入
        self.loss_proj = nn.Sequential(
            nn.Conv2d(4, out_channels, 1),  # 4个损失信号
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()  # 注意力权重
        )
        
    def forward(self, x, prototype_feedback=None, loss_feedback=None):
        # 多尺度空洞卷积
        dilated_1 = self.dilated_conv_1(x)
        dilated_2 = self.dilated_conv_2(x)
        dilated_4 = self.dilated_conv_4(x)
        dilated_8 = self.dilated_conv_8(x)
        
        # 特征融合
        multi_scale = torch.cat([dilated_1, dilated_2, dilated_4, dilated_8], dim=1)
        features = self.fusion_conv(multi_scale)
        
        # 原型反馈
        if prototype_feedback is not None:
            # 上采样原型到当前特征尺寸
            proto_resized = F.interpolate(
                prototype_feedback, 
                size=features.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            proto_att = self.prototype_proj(proto_resized)
            features = features + proto_att
        
        # 损失反馈
        if loss_feedback is not None:
            # 上采样损失信号到当前特征尺寸
            loss_resized = F.interpolate(
                loss_feedback, 
                size=features.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )
            loss_att = self.loss_proj(loss_resized)
            features = features * loss_att
            
        return features


class MSCFE(nn.Module):
    """多尺度对比特征增强模块"""
    def __init__(self, feature_channels, cfpn_channels, out_channels=256):
        super(MSCFE, self).__init__()
        
        # 为B1和Fuse2-4创建MSC块
        self.msc1 = MSCBlock(feature_channels['b1'], out_channels)
        self.msc2 = MSCBlock(cfpn_channels, out_channels)
        self.msc3 = MSCBlock(cfpn_channels, out_channels)
        self.msc4 = MSCBlock(cfpn_channels, out_channels)
        
        # 空洞卷积扩张率
        self.dilation_rates = {
            'e1': 1,
            'e2': 2,
            'e3': 4,
            'e4': 8
        }
        
    def forward(self, encoder_features, cfpn_features, feedback=None):
        """
        Args:
            encoder_features: B1特征
            cfpn_features: CFPN输出的融合特征
            feedback: 包含原型和损失反馈的字典
        """
        # 提取反馈信息
        prototype_feedback = feedback.get('prototypes') if feedback else None
        loss_feedback = feedback.get('losses') if feedback else None
        
        # 应用空洞卷积和MSC处理
        msc1 = self.msc1(
            encoder_features['b1'], 
            prototype_feedback, 
            loss_feedback
        )
        
        msc2 = self.msc2(
            cfpn_features['fuse2'], 
            prototype_feedback, 
            loss_feedback
        )
        
        msc3 = self.msc3(
            cfpn_features['fuse3'], 
            prototype_feedback, 
            loss_feedback
        )
        
        msc4 = self.msc4(
            cfpn_features['fuse4'], 
            prototype_feedback, 
            loss_feedback
        )
        
        return {
            'msc1': msc1,
            'msc2': msc2,
            'msc3': msc3,
            'msc4': msc4
        }
