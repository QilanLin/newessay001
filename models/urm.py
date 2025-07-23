import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyRectifierModule(nn.Module):
    """错误聚焦不确定性矫正模块 (URM)"""
    def __init__(self, prototype_dim=256, temperature_alpha=2.0):
        super(UncertaintyRectifierModule, self).__init__()
        
        self.temperature_alpha = temperature_alpha
        
        # 1. 掩码到logits的转换
        self.mask_to_logits = nn.Conv2d(1, 1, 1)
        
        # 2. 不确定性特征上采样
        self.uncertainty_upsample = nn.Sequential(
            nn.ConvTranspose2d(prototype_dim, prototype_dim // 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(prototype_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(prototype_dim // 2, prototype_dim // 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(prototype_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(prototype_dim // 4, prototype_dim // 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(prototype_dim // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(prototype_dim // 8, prototype_dim, 4, stride=2, padding=1),
            nn.BatchNorm2d(prototype_dim),
            nn.ReLU(inplace=True)
        )
        
        # 注意力生成
        self.attention_gen = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
    def compute_entropy(self, logits):
        """计算像素级熵"""
        # 应用温度缩放
        scaled_logits = logits / self.temperature_alpha
        probs = torch.sigmoid(scaled_logits)
        
        # 避免log(0)
        eps = 1e-8
        probs = torch.clamp(probs, eps, 1 - eps)
        
        # 计算二值熵
        entropy = -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)
        return entropy
        
    def generate_error_mask(self, predicted_mask, ground_truth):
        """生成错误掩码"""
        # 将预测和真值转换为二值
        pred_binary = (predicted_mask > 0.5).float()
        gt_binary = ground_truth.float()
        
        # 错误像素 = 预测与真值不匹配的像素
        error_mask = (pred_binary != gt_binary).float()
        return error_mask
        
    def forward(self, predicted_mask, f_uc, ground_truth=None, training=True):
        """
        Args:
            predicted_mask: 预测掩码 [B, 1, H, W]
            f_uc: 不确定性原型特征 [B, C, H/16, W/16]
            ground_truth: 真值掩码 [B, 1, H, W] (仅训练时需要)
            training: 是否为训练模式
        Returns:
            dict: 包含矫正后的不确定性特征等
        """
        # 1. 掩码转logits
        logits = self.mask_to_logits(predicted_mask)
        
        # 2. 温度缩放和熵计算
        entropy = self.compute_entropy(logits)
        
        # 3. 生成错误掩码和注意力
        error_mask = None
        if training and ground_truth is not None:
            # 训练时：使用真值生成错误掩码
            error_mask = self.generate_error_mask(predicted_mask, ground_truth)
            
            # 结合错误掩码和熵生成注意力
            combined_uncertainty = entropy * error_mask
        else:
            # 推理时：仅使用熵
            combined_uncertainty = entropy
            
        # 4. 生成注意力图
        attention_map = self.attention_gen(combined_uncertainty)
        
        # 5. 不确定性特征上采样（×16）
        f_uc_upsampled = self.uncertainty_upsample(f_uc)
        
        # 确保尺寸匹配
        if f_uc_upsampled.shape[2:] != attention_map.shape[2:]:
            f_uc_upsampled = F.interpolate(
                f_uc_upsampled,
                size=attention_map.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        # 6. Hadamard乘积进行矫正
        f_uc_corrected = f_uc_upsampled * attention_map
        
        return {
            'f_uc_corrected': f_uc_corrected,
            'entropy': entropy,
            'attention_map': attention_map,
            'error_mask': error_mask if training and ground_truth is not None else None
        }
