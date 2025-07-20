import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import EncoderBackbone
from .cfpn import CFPN
from .mscfe import MSCFE
from .decoders import MultiResolutionDecoders
from .feature_decoupling import FeatureDecoupling
from .urm import UncertaintyRectifierModule
from .auxiliary_head import AuxiliaryHead


class EnhancedMedicalSegNet(nn.Module):
    """增强医学图像分割网络"""
    def __init__(
        self, 
        in_channels=3, 
        num_classes=1, 
        base_channels=64, 
        prototype_dim=256,
        temperature_alpha=2.0,
        loss_weights=None
    ):
        super(EnhancedMedicalSegNet, self).__init__()
        
        # 设置默认损失权重
        if loss_weights is None:
            loss_weights = {
                'mask_loss': 1.0,
                'fg_loss': 0.5,
                'bg_loss': 0.5,
                'uncertainty_loss': 0.3
            }
        self.loss_weights = loss_weights
        
        # 1. 编码器主干
        self.encoder = EncoderBackbone(in_channels, base_channels)
        feature_channels = self.encoder.get_feature_channels()
        
        # 2. 对比特征金字塔 (CFPN)
        self.cfpn = CFPN(feature_channels, out_channels=256)
        
        # 3. 多尺度对比特征增强 (MSCFE)
        self.mscfe = MSCFE(feature_channels, cfpn_channels=256, out_channels=256)
        
        # 4. 多分辨率解码器
        self.decoders = MultiResolutionDecoders(msc_channels=256)
        
        # 5. 特征解耦
        self.feature_decoupling = FeatureDecoupling(
            feature_channels['b4'], 
            prototype_dim
        )
        
        # 6. 错误聚焦不确定性矫正 (URM)
        self.urm = UncertaintyRectifierModule(
            prototype_dim, 
            temperature_alpha
        )
        
        # 7. 辅助头
        self.auxiliary_head = AuxiliaryHead(prototype_dim, num_classes=3)
        
        # 损失函数
        self.bce_loss = nn.BCELoss()
        
    def forward(self, x, ground_truth=None, training=True):
        """
        Args:
            x: 输入图像 [B, C, H, W]
            ground_truth: 真值掩码 [B, 1, H, W] (训练时需要)
            training: 是否为训练模式
        Returns:
            如果training=True: 返回总损失
            如果training=False: 返回预测结果字典
        """
        batch_size = x.shape[0]
        
        # 1. 编码器提取特征
        encoder_features = self.encoder(x)
        
        # 2. CFPN处理
        cfpn_features = self.cfpn(encoder_features)
        
        # 3. 特征解耦生成原型
        prototypes = self.feature_decoupling(encoder_features['b4'])
        
        # 初始化反馈为None（第一次前向传播）
        feedback = None
        
        # 4. MSCFE处理（带原型反馈）
        msc_features = self.mscfe(encoder_features, cfpn_features, feedback)
        
        # 5. 多分辨率解码
        decoder_outputs = self.decoders(msc_features)
        predicted_mask = decoder_outputs['mask']
        
        # 6. URM矫正不确定性特征
        urm_outputs = self.urm(
            predicted_mask, 
            prototypes['f_uc'], 
            ground_truth, 
            training
        )
        
        # 更新原型字典
        prototypes['f_uc_corrected'] = urm_outputs['f_uc_corrected']
        
        # 7. 辅助头处理
        aux_outputs = self.auxiliary_head(prototypes)
        
        if training:
            return self._compute_losses(
                predicted_mask, 
                aux_outputs, 
                ground_truth
            )
        else:
            return self._prepare_inference_outputs(
                predicted_mask,
                aux_outputs,
                urm_outputs,
                decoder_outputs
            )
    
    def _compute_losses(self, predicted_mask, aux_outputs, ground_truth):
        """计算所有损失"""
        # 主掩码损失
        loss_mask = self.bce_loss(predicted_mask, ground_truth)
        
        # 辅助损失
        aux_losses = self.auxiliary_head.compute_auxiliary_losses(
            aux_outputs, ground_truth
        )
        
        # 加权总损失
        total_loss = (
            self.loss_weights['mask_loss'] * loss_mask +
            self.loss_weights['fg_loss'] * aux_losses['loss_fg'] +
            self.loss_weights['bg_loss'] * aux_losses['loss_bg'] +
            self.loss_weights['uncertainty_loss'] * aux_losses['loss_compl']
        )
        
        return {
            'total_loss': total_loss,
            'loss_mask': loss_mask,
            'loss_fg': aux_losses['loss_fg'],
            'loss_bg': aux_losses['loss_bg'],
            'loss_compl': aux_losses['loss_compl']
        }
    
    def _prepare_inference_outputs(self, predicted_mask, aux_outputs, urm_outputs, decoder_outputs):
        """准备推理输出"""
        return {
            'mask': predicted_mask,
            'foreground_prob': aux_outputs['u_fg'],
            'background_prob': aux_outputs['u_bg'],
            'uncertainty_prob': aux_outputs['u_compl'],
            'entropy': urm_outputs['entropy'],
            'attention_map': urm_outputs['attention_map'],
            'decoder_outputs': decoder_outputs
        }
    
    def enable_feedback_training(self):
        """启用原型和损失反馈训练（高级训练模式）"""
        # 这个方法可以在未来扩展，实现更复杂的反馈机制
        pass
    
    def get_model_summary(self):
        """返回模型结构摘要"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_components': [
                'EncoderBackbone',
                'CFPN', 
                'MSCFE',
                'MultiResolutionDecoders',
                'FeatureDecoupling',
                'UncertaintyRectifierModule',
                'AuxiliaryHead'
            ]
        }


# 便捷的模型创建函数
def create_enhanced_medical_seg_net(config=None):
    """创建增强医学分割网络的便捷函数"""
    if config is None:
        # 默认配置
        config = {
            'in_channels': 3,
            'num_classes': 1,
            'base_channels': 64,
            'prototype_dim': 256,
            'temperature_alpha': 2.0,
            'loss_weights': {
                'mask_loss': 1.0,
                'fg_loss': 0.5,
                'bg_loss': 0.5,
                'uncertainty_loss': 0.3
            }
        }
    
    model = EnhancedMedicalSegNet(**config)
    return model
