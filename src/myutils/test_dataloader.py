import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch



class UIEBD_Dataset(Dataset):
    def __init__(self, path_dir,img_size=256):
        super().__init__()
        self.path_dir = path_dir # degraduation images
        self.img_size = img_size

        # 加载所有图像ID（仅加载一次）
        self.image_ids = self._load_image_ids()
        print(f"Total loaded images: {len(self.image_ids)}")

        # 数据变换
        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(self.img_size),
        ])
        self.to_tensor = ToTensor()

    def _load_image_ids(self):
        """一次性加载所有图像ID，不区分类型"""
        # 获取退化图像目录下的所有文件
        degrad_files = [f for f in os.listdir(self.path_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]



        return degrad_files

    def _resize_small_image(self, img):

        # 用PIL缩放（保持高质量）
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize((self.img_size,self.img_size), Image.LANCZOS)  # 抗锯齿缩放
        return np.array(img_resized)


    def __getitem__(self, idx):
        """获取数据样本，不区分退化类型"""
        sample = self.image_ids[idx]

        # 加载退化图像和清晰图像（仅在获取样本时加载，避免内存占用过高）
        degrad_path = os.path.join(self.path_dir, sample)

        # 读取图像并转换为RGB
        degrad_img = np.array(Image.open(degrad_path).convert('RGB'))

        # 对小尺寸图像进行缩放
        degrad_img = self._resize_small_image(degrad_img)
        # 转回numpy数组（确保后续to_tensor能处理）
        degrad_patch = np.array(degrad_img)

        # 转换为Tensor
        degrad_tensor = self.to_tensor(degrad_patch)

        # 返回文件名（用于调试）和处理后的张量
        return sample, degrad_tensor

    def __len__(self):
        return len(self.image_ids)
