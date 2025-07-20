import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os

from models.enhanced_medical_seg import create_enhanced_medical_seg_net


class MedicalSegInference:
    """医学图像分割推理器"""
    def __init__(self, checkpoint_path, config=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 使用检查点中的配置或提供的配置
        model_config = config or checkpoint.get('config', {}).get('model', {})
        
        # 创建模型
        self.model = create_enhanced_medical_seg_net(model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded from {checkpoint_path}")
        print(f"Inference on device: {self.device}")
    
    def preprocess_image(self, image_path, target_size=(512, 512)):
        """预处理输入图像"""
        # 读取图像
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # 调整大小
        image = image.resize(target_size, Image.BILINEAR)
        
        # 转换为numpy数组并归一化
        image_array = np.array(image, dtype=np.float32) / 255.0
        
        # 转换为tensor并添加batch维度
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def postprocess_mask(self, mask_tensor, threshold=0.5):
        """后处理预测掩码"""
        # 移除batch维度并转换为numpy
        mask = mask_tensor.squeeze().cpu().numpy()
        
        # 二值化
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        
        return mask, binary_mask
    
    def predict(self, image_path, return_all_outputs=False):
        """对单张图像进行预测"""
        with torch.no_grad():
            # 预处理
            image_tensor = self.preprocess_image(image_path)
            
            # 推理
            outputs = self.model(image_tensor, training=False)
            
            # 后处理主掩码
            mask_prob, binary_mask = self.postprocess_mask(outputs['mask'])
            
            results = {
                'mask_probability': mask_prob,
                'binary_mask': binary_mask,
            }
            
            if return_all_outputs:
                # 添加所有输出
                fg_prob, _ = self.postprocess_mask(outputs['foreground_prob'])
                bg_prob, _ = self.postprocess_mask(outputs['background_prob'])
                uc_prob, _ = self.postprocess_mask(outputs['uncertainty_prob'])
                entropy, _ = self.postprocess_mask(outputs['entropy'])
                attention, _ = self.postprocess_mask(outputs['attention_map'])
                
                results.update({
                    'foreground_probability': fg_prob,
                    'background_probability': bg_prob,
                    'uncertainty_probability': uc_prob,
                    'entropy_map': entropy,
                    'attention_map': attention,
                    'raw_outputs': outputs
                })
            
            return results
    
    def predict_batch(self, image_paths, batch_size=4):
        """批量预测"""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_results = []
            
            with torch.no_grad():
                # 预处理批次
                batch_tensors = []
                for path in batch_paths:
                    tensor = self.preprocess_image(path)
                    batch_tensors.append(tensor)
                
                batch_input = torch.cat(batch_tensors, dim=0)
                
                # 批量推理
                outputs = self.model(batch_input, training=False)
                
                # 处理每个样本的结果
                for j in range(len(batch_paths)):
                    mask_prob, binary_mask = self.postprocess_mask(outputs['mask'][j:j+1])
                    batch_results.append({
                        'image_path': batch_paths[j],
                        'mask_probability': mask_prob,
                        'binary_mask': binary_mask
                    })
            
            results.extend(batch_results)
        
        return results
    
    def visualize_results(self, image_path, results, save_path=None):
        """可视化预测结果"""
        # 读取原始图像
        original_image = Image.open(image_path).convert('RGB')
        original_image = original_image.resize((512, 512))
        
        # 创建子图
        if 'foreground_probability' in results:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # 原始图像
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 预测掩码
        axes[1].imshow(results['mask_probability'], cmap='hot')
        axes[1].set_title('Predicted Mask (Probability)')
        axes[1].axis('off')
        
        # 二值掩码
        axes[2].imshow(results['binary_mask'], cmap='gray')
        axes[2].set_title('Binary Mask')
        axes[2].axis('off')
        
        # 如果有详细输出，显示更多结果
        if 'foreground_probability' in results:
            axes[3].imshow(results['foreground_probability'], cmap='Reds')
            axes[3].set_title('Foreground Probability')
            axes[3].axis('off')
            
            axes[4].imshow(results['background_probability'], cmap='Blues')
            axes[4].set_title('Background Probability')
            axes[4].axis('off')
            
            axes[5].imshow(results['uncertainty_probability'], cmap='Purples')
            axes[5].set_title('Uncertainty Probability')
            axes[5].axis('off')
            
            axes[6].imshow(results['entropy_map'], cmap='viridis')
            axes[6].set_title('Entropy Map')
            axes[6].axis('off')
            
            axes[7].imshow(results['attention_map'], cmap='plasma')
            axes[7].set_title('Attention Map')
            axes[7].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Results saved to {save_path}")
        
        plt.show()
    
    def save_mask(self, mask, save_path):
        """保存掩码为图像文件"""
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        cv2.imwrite(save_path, mask)
        print(f"Mask saved to {save_path}")


def main():
    """示例推理脚本"""
    # 配置路径
    checkpoint_path = 'checkpoints/best_model.pth'
    image_path = 'test_image.jpg'  # 替换为您的测试图像路径
    
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first or provide correct checkpoint path.")
        return
    
    if not os.path.exists(image_path):
        print(f"Test image not found: {image_path}")
        print("Please provide a valid image path.")
        return
    
    # 创建推理器
    inferencer = MedicalSegInference(checkpoint_path)
    
    # 进行预测
    print("Performing inference...")
    results = inferencer.predict(image_path, return_all_outputs=True)
    
    # 可视化结果
    inferencer.visualize_results(
        image_path, 
        results, 
        save_path='inference_results.png'
    )
    
    # 保存掩码
    inferencer.save_mask(
        results['binary_mask'], 
        'predicted_mask.png'
    )
    
    print("Inference completed!")


if __name__ == '__main__':
    main()
