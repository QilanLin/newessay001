"""
医学图像数据加载器示例
支持常见的医学图像分割数据集格式
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MedicalImageDataset(Dataset):
    """医学图像分割数据集"""
    
    def __init__(self, image_dir, mask_dir, image_size=(512, 512), transforms=None):
        """
        Args:
            image_dir: 图像文件夹路径
            mask_dir: 掩码文件夹路径
            image_size: 目标图像尺寸
            transforms: 数据增强变换
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.transforms = transforms
        
        # 获取所有图像文件名
        self.image_files = sorted([f for f in os.listdir(image_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))])
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 获取图像和掩码路径
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        
        # 掩码文件名（通常与图像同名或有特定后缀）
        mask_name = self.get_mask_name(image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 读取图像和掩码
        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)
        
        # 应用变换
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            # 当未指定变换时，将numpy数组转换为张量
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, 0)
            image = torch.from_numpy(image).permute(2, 0, 1)
            mask = torch.from_numpy(mask)
        
        # 确保掩码是正确的格式
        if len(mask.shape) == 3:
            mask = mask[0:1]  # 取第一个通道
        elif len(mask.shape) == 2:
            mask = mask.unsqueeze(0)  # 添加通道维度
        
        return image.float(), mask.float()
    
    def get_mask_name(self, image_name):
        """根据图像名获取对应的掩码名"""
        name, ext = os.path.splitext(image_name)
        
        # 常见的掩码命名规则
        possible_names = [
            f"{name}_mask{ext}",
            f"{name}_gt{ext}",
            f"{name}_label{ext}",
            f"{name}.png",  # 掩码通常是PNG格式
            image_name  # 同名文件
        ]
        
        for mask_name in possible_names:
            if os.path.exists(os.path.join(self.mask_dir, mask_name)):
                return mask_name
        
        # 如果都找不到，返回默认名称
        return f"{name}.png"
    
    def load_image(self, image_path):
        """加载图像"""
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.image_size, Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        return image
    
    def load_mask(self, mask_path):
        """加载掩码"""
        if not os.path.exists(mask_path):
            # 如果掩码不存在，创建零掩码
            print(f"Warning: Mask not found at {mask_path}, creating zero mask")
            mask = np.zeros(self.image_size, dtype=np.float32)
        else:
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize(self.image_size, Image.NEAREST)
            mask = np.array(mask, dtype=np.float32)
            # 二值化掩码
            mask = (mask > 128).astype(np.float32)
        
        return mask


def get_train_transforms(image_size=(512, 512)):
    """获取训练时的数据增强"""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_val_transforms(image_size=(512, 512)):
    """获取验证时的数据变换"""
    return A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def create_dataloaders(
    train_image_dir,
    train_mask_dir,
    val_image_dir=None,
    val_mask_dir=None,
    batch_size=8,
    image_size=(512, 512),
    num_workers=4,
    val_split=0.2
):
    """创建训练和验证数据加载器"""
    
    # 训练数据集
    train_dataset = MedicalImageDataset(
        train_image_dir,
        train_mask_dir,
        image_size,
        get_train_transforms(image_size)
    )
    
    # 如果没有单独的验证集，从训练集中分割
    if val_image_dir is None:
        total_size = len(train_dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        
        # 为验证集设置不同的变换
        val_dataset.dataset.transforms = get_val_transforms(image_size)
    else:
        # 使用单独的验证集
        val_dataset = MedicalImageDataset(
            val_image_dir,
            val_mask_dir,
            image_size,
            get_val_transforms(image_size)
        )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# 示例数据集格式
def print_dataset_format():
    """打印期望的数据集格式"""
    print("=== 数据集格式说明 ===")
    print("""
期望的文件夹结构:

dataset/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── masks/
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
└── val/
    ├── images/
    │   ├── val_001.jpg
    │   └── ...
    └── masks/
        ├── val_001.png
        └── ...

说明:
1. 图像格式: JPG, PNG, TIFF等
2. 掩码格式: PNG (推荐), 二值图像 (0-背景, 255-前景)
3. 文件名: 图像和对应掩码应有相同的基本文件名
4. 尺寸: 任意尺寸 (会自动调整到指定大小)
    """)


# 测试数据加载器
def test_dataloader():
    """测试数据加载器功能"""
    print("=== 测试数据加载器 ===")
    
    # 创建模拟数据集路径
    train_image_dir = "dataset/train/images"
    train_mask_dir = "dataset/train/masks"
    
    if not os.path.exists(train_image_dir):
        print(f"数据集路径不存在: {train_image_dir}")
        print("请准备数据集或修改路径")
        return
    
    try:
        # 创建数据加载器
        train_loader, val_loader = create_dataloaders(
            train_image_dir,
            train_mask_dir,
            batch_size=2,
            image_size=(256, 256)
        )
        
        print(f"训练集批次数: {len(train_loader)}")
        print(f"验证集批次数: {len(val_loader)}")
        
        # 测试一个批次
        for images, masks in train_loader:
            print(f"图像批次形状: {images.shape}")
            print(f"掩码批次形状: {masks.shape}")
            print(f"图像值范围: [{images.min():.3f}, {images.max():.3f}]")
            print(f"掩码值范围: [{masks.min():.3f}, {masks.max():.3f}]")
            break
            
    except Exception as e:
        print(f"数据加载器测试失败: {e}")


def main():
    """主函数"""
    print("医学图像数据加载器")
    print("=" * 30)
    
    print_dataset_format()
    test_dataloader()
    
    print("\n=== 使用示例 ===")
    print("""
# 创建数据加载器
train_loader, val_loader = create_dataloaders(
    'dataset/train/images',
    'dataset/train/masks',
    batch_size=8,
    image_size=(512, 512)
)

# 在训练循环中使用
for epoch in range(num_epochs):
    for images, masks in train_loader:
        # 训练代码
        pass
    """)


if __name__ == '__main__':
    main()
