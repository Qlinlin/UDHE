import os
import random
import copy
from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor
import torch

from myutils.image_utils import random_augmentation
# Python随机库
random.seed(42)
# NumPy
np.random.seed(42)
# PyTorch CPU
torch.manual_seed(42)
# PyTorch GPU（如果使用）
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class UIEBD_Dataset(Dataset):
    def __init__(self, path_dir,img_size,mode='train'):
        super().__init__()
        self.path_dir = path_dir
        self.img_size = img_size
        self.mode = mode
        # 图像路径配置（仅使用单一数据目录结构）
        self.degrad_dir = os.path.join(self.path_dir, 'input')  # 退化图像目录
        self.clean_dir = os.path.join(self.path_dir, 'target')  # 清晰图像目录

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
        degrad_files = [f for f in os.listdir(self.degrad_dir)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        # 验证对应的清晰图像是否存在
        valid_ids = []
        for filename in degrad_files:
            clean_path = os.path.join(self.clean_dir, filename)
            if os.path.exists(clean_path):
                valid_ids.append({
                    'degrad_id': filename,  # 退化图像文件名
                    'clean_id': filename  # 清晰图像文件名（假设同名）
                })

       

        return valid_ids

    def _resize_small_image(self, img):
        """对小于目标尺寸的图像进行等比例缩放，确保最小边 >= img_size"""
        h, w = img.shape[:2]
        if h >= self.img_size and w >= self.img_size:
            return img  # 尺寸足够，无需缩放

        # 计算缩放比例（保证最小边达到目标尺寸）
        scale = max((self.img_size+10) / h, (self.img_size+10) / w)
        new_h = int(h * scale)
        new_w = int(w * scale)

        # 用PIL缩放（保持高质量）
        img_pil = Image.fromarray(img)
        img_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)  # 抗锯齿缩放
        return np.array(img_resized)

    def _crop_patch(self, img1, img2):
        """裁剪配对的图像块"""
        H, W = img1.shape[:2]
        ind_H = random.randint(0, H - self.img_size)
        ind_W = random.randint(0, W - self.img_size)

        patch1 = img1[ind_H:ind_H + self.img_size, ind_W:ind_W + self.img_size]
        patch2 = img2[ind_H:ind_H + self.img_size, ind_W:ind_W + self.img_size]
        return patch1, patch2

    def __getitem__(self, idx):
        """获取数据样本，不区分退化类型"""
        sample = self.image_ids[idx]

        # 加载退化图像和清晰图像（仅在获取样本时加载，避免内存占用过高）
        degrad_path = os.path.join(self.degrad_dir, sample['degrad_id'])
        clean_path = os.path.join(self.clean_dir, sample['clean_id'])

        # 读取图像并转换为RGB
        degrad_img = np.array(Image.open(degrad_path).convert('RGB'))
        clean_img = np.array(Image.open(clean_path).convert('RGB'))

        # 关键步骤：对小尺寸图像进行缩放
        degrad_img = self._resize_small_image(degrad_img)
        clean_img = self._resize_small_image(clean_img)  # 同步缩放配对图像
        if self.mode == 'train':
           
            degrad_pil = Image.fromarray(degrad_img)
            clean_pil = Image.fromarray(clean_img)
            # 按目标尺寸缩放
            degrad_patch = degrad_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
            clean_patch = clean_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
            # 转回numpy数组
            degrad_patch = np.array(degrad_patch)
            clean_patch = np.array(clean_patch)
            # 随机数据增强（假设random_augmentation是已定义的增强函数）
            degrad_patch, clean_patch = random_augmentation(degrad_patch, clean_patch)
        else:
            # 验证集处理：正确调整尺寸
            # 将numpy数组转为PIL Image
            degrad_pil = Image.fromarray(degrad_img)
            clean_pil = Image.fromarray(clean_img)
            # 按目标尺寸缩放
            degrad_patch = degrad_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
            clean_patch = clean_pil.resize((self.img_size, self.img_size), Image.LANCZOS)
            # 转回numpy数组
            degrad_patch = np.array(degrad_patch)
            clean_patch = np.array(clean_patch)

        # 转换为Tensor
        degrad_tensor = self.to_tensor(degrad_patch)
        clean_tensor = self.to_tensor(clean_patch)

        # 返回文件名和处理后的张量
        return [sample['clean_id'], 0], degrad_tensor, clean_tensor  # 0表示不区分类型

    def __len__(self):
        return len(self.image_ids)