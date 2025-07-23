import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import os
from tqdm import tqdm

from models.enhanced_medical_seg import create_enhanced_medical_seg_net


class MedicalSegTrainer:
    """医学图像分割训练器"""
    def __init__(self, config_path='configs/config.yaml'):
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建模型，同时合并自定义损失权重
        model_config = {
            **self.config['model'],
            'loss_weights': self.config.get('loss_weights')
        }
        self.model = create_enhanced_medical_seg_net(model_config)
        self.model.to(self.device)
        
        # 创建优化器
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        print(f"Model created with {self.model.get_model_summary()['total_parameters']:,} parameters")
        print(f"Training on device: {self.device}")
        
    def load_config(self, config_path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    
    def create_optimizer(self):
        """创建优化器"""
        if self.config['optimizer']['type'] == 'Adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['optimizer']['type'] == 'SGD':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=self.config['optimizer']['momentum'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']['type']}")
    
    def create_scheduler(self):
        """创建学习率调度器"""
        if self.config['optimizer']['scheduler'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['training']['epochs']
            )
        elif self.config['optimizer']['scheduler'] == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=50, 
                gamma=0.5
            )
        else:
            return None
    
    def train_epoch(self, dataloader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        loss_components = {'mask': 0.0, 'fg': 0.0, 'bg': 0.0, 'compl': 0.0}
        
        pbar = tqdm(dataloader, desc='Training')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            losses = self.model(images, masks, training=True)
            
            # 反向传播
            losses['total_loss'].backward()
            self.optimizer.step()
            
            # 记录损失
            total_loss += losses['total_loss'].item()
            loss_components['mask'] += losses['loss_mask'].item()
            loss_components['fg'] += losses['loss_fg'].item()
            loss_components['bg'] += losses['loss_bg'].item()
            loss_components['compl'] += losses['loss_compl'].item()
            
            # 更新进度条
            pbar.set_postfix({
                'Total Loss': f"{losses['total_loss'].item():.4f}",
                'Mask Loss': f"{losses['loss_mask'].item():.4f}",
                'LR': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        avg_components = {k: v / len(dataloader) for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def validate_epoch(self, dataloader):
        """验证一个epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # 前向传播
                losses = self.model(images, masks, training=True)
                total_loss += losses['total_loss'].item()
                
                pbar.set_postfix({
                    'Val Loss': f"{losses['total_loss'].item():.4f}"
                })
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def train(self, train_dataloader, val_dataloader=None, save_dir='checkpoints'):
        """完整训练流程"""
        os.makedirs(save_dir, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['training']['epochs']}")
            
            # 训练
            train_loss, loss_components = self.train_epoch(train_dataloader)
            
            # 验证
            val_loss = None
            if val_dataloader is not None:
                val_loss = self.validate_epoch(val_dataloader)
                print(f"Validation Loss: {val_loss:.4f}")
                
                # 保存最佳模型
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(
                        os.path.join(save_dir, 'best_model.pth'),
                        epoch, train_loss, val_loss
                    )
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 打印训练结果
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Loss Components - Mask: {loss_components['mask']:.4f}, "
                  f"FG: {loss_components['fg']:.4f}, "
                  f"BG: {loss_components['bg']:.4f}, "
                  f"Uncertainty: {loss_components['compl']:.4f}")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth'),
                    epoch, train_loss, val_loss
                )
    
    def save_checkpoint(self, filepath, epoch, train_loss, val_loss=None):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': self.config
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {filepath}")
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint.get('val_loss')


def main():
    """主训练函数"""
    # 创建训练器
    trainer = MedicalSegTrainer('configs/config.yaml')
    
    # 这里需要实现您的数据加载器
    # train_dataloader = create_dataloaders(...)
    # val_dataloader = create_dataloaders(...)
    
    print("请实现数据加载器后开始训练")
    print("训练器已准备就绪，模型架构如下：")
    print(trainer.model.get_model_summary())


if __name__ == '__main__':
    main()
