import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxiliaryHead(nn.Module):
    """辅助头 + 不确定性分类"""
    def __init__(self, prototype_dim=256, num_classes=3):
        super(AuxiliaryHead, self).__init__()
        
        # 特征对齐层
        self.align_fg = nn.Sequential(
            nn.Conv2d(prototype_dim, prototype_dim, 1),
            nn.BatchNorm2d(prototype_dim),
            nn.ReLU(inplace=True)
        )
        
        self.align_bg = nn.Sequential(
            nn.Conv2d(prototype_dim, prototype_dim, 1),
            nn.BatchNorm2d(prototype_dim),
            nn.ReLU(inplace=True)
        )
        
        self.align_uc = nn.Sequential(
            nn.Conv2d(prototype_dim, prototype_dim, 1),
            nn.BatchNorm2d(prototype_dim),
            nn.ReLU(inplace=True)
        )
        
        # 三分类头（前景、背景、不确定性）
        self.classifier = nn.Sequential(
            nn.Conv2d(prototype_dim * 3, prototype_dim, 1),
            nn.BatchNorm2d(prototype_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(prototype_dim, num_classes, 1)
        )
        
        # Softmax用于生成概率
        self.softmax = nn.Softmax(dim=1)
        
        self.num_classes = num_classes
        
    def forward(self, prototypes):
        """
        Args:
            prototypes: 包含f_fg, f_bg, f_uc_corrected的字典
        Returns:
            dict: 包含各类别概率和损失的字典
        """
        f_fg = prototypes['f_fg']
        f_bg = prototypes['f_bg']
        f_uc_corrected = prototypes['f_uc_corrected']
        
        # 特征对齐
        aligned_fg = self.align_fg(f_fg)
        aligned_bg = self.align_bg(f_bg)
        aligned_uc = self.align_uc(f_uc_corrected)
        
        # 确保所有特征尺寸一致
        target_size = aligned_fg.shape[2:]
        if aligned_bg.shape[2:] != target_size:
            aligned_bg = F.interpolate(
                aligned_bg, size=target_size, mode='bilinear', align_corners=False
            )
        if aligned_uc.shape[2:] != target_size:
            aligned_uc = F.interpolate(
                aligned_uc, size=target_size, mode='bilinear', align_corners=False
            )
        
        # 特征串接
        combined_features = torch.cat([aligned_fg, aligned_bg, aligned_uc], dim=1)
        
        # 分类
        logits = self.classifier(combined_features)
        probs = self.softmax(logits)
        
        # 分离各类别概率
        u_fg = probs[:, 0:1, :, :]      # 前景概率
        u_bg = probs[:, 1:2, :, :]      # 背景概率  
        u_compl = probs[:, 2:3, :, :]   # 不确定性概率
        
        return {
            'logits': logits,
            'u_fg': u_fg,
            'u_bg': u_bg,
            'u_compl': u_compl,
            'combined_probs': probs
        }
    
    def compute_auxiliary_losses(self, outputs, ground_truth):
        """计算辅助损失"""
        # 生成伪标签
        gt_binary = (ground_truth > 0.5).float()
        
        # 前景损失：真值为1的区域
        fg_target = gt_binary
        loss_fg = F.binary_cross_entropy(outputs['u_fg'], fg_target)
        
        # 背景损失：真值为0的区域
        bg_target = 1.0 - gt_binary
        loss_bg = F.binary_cross_entropy(outputs['u_bg'], bg_target)
        
        # 不确定性损失：边界区域或模糊区域
        # 这里使用梯度幅度来定义不确定性区域
        uncertainty_target = self.generate_uncertainty_target(ground_truth)
        loss_compl = F.binary_cross_entropy(outputs['u_compl'], uncertainty_target)
        
        return {
            'loss_fg': loss_fg,
            'loss_bg': loss_bg,
            'loss_compl': loss_compl
        }
    
    def generate_uncertainty_target(self, ground_truth):
        """生成不确定性目标"""
        # 使用Sobel算子检测边界
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=torch.float32, device=ground_truth.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=torch.float32, device=ground_truth.device).view(1, 1, 3, 3)
        
        # 计算梯度
        grad_x = F.conv2d(ground_truth, sobel_x, padding=1)
        grad_y = F.conv2d(ground_truth, sobel_y, padding=1)
        
        # 梯度幅度
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # 阈值化生成不确定性目标
        uncertainty_target = (gradient_magnitude > 0.1).float()
        
        return uncertainty_target
