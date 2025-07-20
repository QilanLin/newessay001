# Enhanced Medical Image Segmentation Network

基于对比特征金字塔和错误聚焦不确定性矫正的增强医学图像分割网络。

## 网络架构

该网络包含以下主要组件：

1. **编码器主干** - 四个顺序卷积块，整体步幅为16
2. **对比特征金字塔 (CFPN)** - 多尺度特征融合
3. **多尺度对比特征增强 (MSCFE)** - 空洞卷积增强
4. **多分辨率解码器** - 四个不同尺度的解码器
5. **特征解耦** - 前景、背景和不确定性特征分离
6. **错误聚焦不确定性矫正 (URM)** - 基于熵的不确定性矫正
7. **辅助头** - 多任务学习和原型反馈

## 使用方法

```python
from models.enhanced_medical_seg import EnhancedMedicalSegNet

# 创建模型
model = EnhancedMedicalSegNet(
    in_channels=3,
    num_classes=1,
    base_channels=64,
    prototype_dim=256
)

# 训练
loss = model(images, masks)

# 推理
with torch.no_grad():
    outputs = model(images, training=False)
    predictions = outputs['mask']
```

## 文件结构

```
models/
├── __init__.py
├── enhanced_medical_seg.py    # 主网络模型
├── encoder.py                 # 编码器主干
├── cfpn.py                   # 对比特征金字塔
├── mscfe.py                  # 多尺度对比特征增强
├── decoders.py               # 解码器模块
├── feature_decoupling.py     # 特征解耦
├── urm.py                    # 不确定性矫正模块
└── auxiliary_head.py         # 辅助头
```
