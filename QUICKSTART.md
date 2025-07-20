# 快速开始指南

## 1. 环境准备

### 安装依赖
```bash
pip install -r requirements.txt
```

### 系统要求
- Python 3.7+
- PyTorch 1.9+
- CUDA 10.2+ (推荐GPU训练)
- 内存 16GB+ (建议)

## 2. 数据准备

### 数据集格式
```
dataset/
├── train/
│   ├── images/          # 训练图像
│   └── masks/           # 训练掩码
└── val/
    ├── images/          # 验证图像  
    └── masks/           # 验证掩码
```

### 支持的格式
- 图像: JPG, PNG, TIFF
- 掩码: PNG (二值图像, 0-背景, 255-前景)

## 3. 配置调整

编辑 `configs/config.yaml`:

```yaml
# 基本配置
model:
  in_channels: 3        # RGB=3, 灰度=1
  base_channels: 64     # 基础通道数
  prototype_dim: 256    # 原型维度

# 训练配置  
training:
  batch_size: 8         # 根据GPU内存调整
  learning_rate: 0.001
  epochs: 200

# 数据配置
data:
  image_size: [512, 512]  # 目标图像尺寸
```

## 4. 模型测试

```bash
# 测试模型结构
python test_model.py

# 查看使用示例
python examples.py
```

## 5. 开始训练

```bash
# 修改train.py中的数据路径
# 然后运行训练
python train.py
```

## 6. 模型推理

```bash
# 修改inference.py中的模型和图像路径
python inference.py
```

## 7. 自定义使用

### 创建模型
```python
from models.enhanced_medical_seg import create_enhanced_medical_seg_net

# 默认配置
model = create_enhanced_medical_seg_net()

# 自定义配置
config = {
    'in_channels': 1,
    'base_channels': 32,
    'prototype_dim': 128
}
model = create_enhanced_medical_seg_net(config)
```

### 训练
```python
# 训练模式
loss_dict = model(images, masks, training=True)
total_loss = loss_dict['total_loss']
total_loss.backward()
```

### 推理
```python
# 推理模式
model.eval()
with torch.no_grad():
    outputs = model(images, training=False)
    predicted_mask = outputs['mask']
```

## 8. 模型特点

- ✅ 多尺度特征提取与融合
- ✅ 对比学习增强特征表示  
- ✅ 不确定性建模与矫正
- ✅ 多任务学习(分割+分类)
- ✅ 原型与损失反馈机制
- ✅ 错误聚焦的自适应优化

## 9. 故障排除

### 常见问题

**内存不足**
- 减小batch_size
- 降低image_size  
- 减少base_channels

**训练不收敛**
- 检查数据标注质量
- 调整学习率
- 调整损失权重

**推理速度慢**
- 使用GPU
- 减小模型尺寸
- 使用模型量化

### 性能优化

**GPU训练**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

**混合精度训练**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    loss = model(images, masks, training=True)['total_loss']
```

**数据并行**
```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 10. 进阶使用

### 损失权重调整
```python
loss_weights = {
    'mask_loss': 1.0,      # 主分割损失
    'fg_loss': 0.5,        # 前景损失
    'bg_loss': 0.5,        # 背景损失  
    'uncertainty_loss': 0.3 # 不确定性损失
}
```

### 模型分析
```python
summary = model.get_model_summary()
print(f"参数量: {summary['total_parameters']:,}")
```

### 特征可视化
```python
# 获取中间特征进行分析
outputs = model(images, training=False)
entropy_map = outputs['entropy']
attention_map = outputs['attention_map']
```

---

🎉 **恭喜！您已成功设置增强医学图像分割网络！**

如有问题，请查看示例代码或检查配置文件。
