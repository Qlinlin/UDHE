import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision


# 彩色转黑白
def rgb_to_gray(rgb):
    return 0.299 * rgb[:, 0:1, :, :] + 0.587 * rgb[:, 1:2, :, :] + 0.114 * rgb[:, 2:3, :, :]

if __name__ == '__main__':
    
    img = cv2.imread("val/input/41_img_.png",cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_t = torchvision.transforms.ToTensor()(img_rgb)
    print(img_t.shape)
    img_gray = rgb_to_gray(img_t.unsqueeze(0)).squeeze(0).permute(1,2,0)
    plt.imshow(img_gray,cmap='gray')
    plt.show()
