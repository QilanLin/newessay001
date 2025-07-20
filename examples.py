"""
增强医学图像分割网络使用示例
演示如何使用模型进行训练和推理
"""

# 示例1: 基本使用
def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===")
    print("""
# 1. 导入模型
from models.enhanced_medical_seg import create_enhanced_medical_seg_net

# 2. 创建模型
model = create_enhanced_medical_seg_net()

# 3. 训练模式
model.train()
loss_dict = model(images, ground_truth_masks, training=True)
total_loss = loss_dict['total_loss']

# 4. 推理模式
model.eval()
with torch.no_grad():
    outputs = model(images, training=False)
    predicted_mask = outputs['mask']
    """)


# 示例2: 自定义配置
def example_custom_config():
    """自定义配置示例"""
    print("=== 自定义配置示例 ===")
    print("""
# 自定义模型配置
config = {
    'in_channels': 1,      # 灰度图像
    'num_classes': 1,      # 二分类分割
    'base_channels': 32,   # 较小的模型
    'prototype_dim': 128,  # 原型维度
    'temperature_alpha': 1.5,
    'loss_weights': {
        'mask_loss': 1.0,
        'fg_loss': 0.3,
        'bg_loss': 0.3,
        'uncertainty_loss': 0.2
    }
}

model = create_enhanced_medical_seg_net(config)
    """)


# 示例3: 训练循环
def example_training_loop():
    """训练循环示例"""
    print("=== 训练循环示例 ===")
    print("""
import torch.optim as optim

# 创建模型和优化器
model = create_enhanced_medical_seg_net()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(num_epochs):
    for batch_idx, (images, masks) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # 前向传播
        loss_dict = model(images, masks, training=True)
        total_loss = loss_dict['total_loss']
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        # 打印损失
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {total_loss.item():.4f}')
    """)


# 示例4: 推理和可视化
def example_inference():
    """推理示例"""
    print("=== 推理示例 ===")
    print("""
# 加载训练好的模型
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 推理单张图像
with torch.no_grad():
    outputs = model(test_image, training=False)
    
    # 获取各种输出
    mask = outputs['mask']                    # 主分割掩码
    fg_prob = outputs['foreground_prob']      # 前景概率
    bg_prob = outputs['background_prob']      # 背景概率  
    uncertainty = outputs['uncertainty_prob'] # 不确定性概率
    entropy = outputs['entropy']              # 熵图
    attention = outputs['attention_map']      # 注意力图

# 后处理
predicted_mask = (mask > 0.5).float()
    """)


# 示例5: 损失分析
def example_loss_analysis():
    """损失分析示例"""
    print("=== 损失分析示例 ===")
    print("""
# 训练时获取详细损失信息
loss_dict = model(images, masks, training=True)

print(f"总损失: {loss_dict['total_loss']:.4f}")
print(f"掩码损失: {loss_dict['loss_mask']:.4f}")
print(f"前景损失: {loss_dict['loss_fg']:.4f}")  
print(f"背景损失: {loss_dict['loss_bg']:.4f}")
print(f"不确定性损失: {loss_dict['loss_compl']:.4f}")

# 分析各损失分量的贡献
total = loss_dict['total_loss'].item()
mask_contrib = loss_dict['loss_mask'].item() / total * 100
fg_contrib = loss_dict['loss_fg'].item() / total * 100
bg_contrib = loss_dict['loss_bg'].item() / total * 100
uc_contrib = loss_dict['loss_compl'].item() / total * 100

print(f"损失贡献比例:")
print(f"  掩码: {mask_contrib:.1f}%")
print(f"  前景: {fg_contrib:.1f}%") 
print(f"  背景: {bg_contrib:.1f}%")
print(f"  不确定性: {uc_contrib:.1f}%")
    """)


# 示例6: 模型分析
def example_model_analysis():
    """模型分析示例"""
    print("=== 模型分析示例 ===")
    print("""
# 获取模型信息
summary = model.get_model_summary()
print(f"总参数量: {summary['total_parameters']:,}")
print(f"可训练参数: {summary['trainable_parameters']:,}")
print(f"模型组件: {summary['model_components']}")

# 计算模型复杂度
def count_flops(model, input_size):
    # 使用 thop 或 fvcore 库计算 FLOPs
    pass

# 分析中间特征
def analyze_features(model, input_tensor):
    # Hook方式获取中间特征进行分析
    pass
    """)


def main():
    """主函数"""
    print("增强医学图像分割网络 - 使用示例")
    print("=" * 50)
    
    example_basic_usage()
    example_custom_config()
    example_training_loop()
    example_inference()
    example_loss_analysis()
    example_model_analysis()
    
    print("\n=== 重要提示 ===")
    print("1. 确保安装所有依赖: pip install -r requirements.txt")
    print("2. 准备医学图像数据集")
    print("3. 根据数据集调整配置文件")
    print("4. 使用GPU训练以获得更好性能")
    print("5. 监控训练过程中的各项损失")
    print("\n模型实现完成！可以开始使用了。")


if __name__ == '__main__':
    main()
