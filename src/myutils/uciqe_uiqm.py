import os
from glob import glob

import cv2

from uciqe import uciqe
from uiqm import getUIQM


def calculate_folder_uciqe_uiqm_average(folder_path):
    """
    计算文件夹中所有图片的UIQM平均值
    :param folder_path: 图片文件夹路径
    :return: 平均值和有效图片数量
    """
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
    image_paths = []

    # 收集所有图片路径
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(folder_path, ext)))


    total_uciqe = 0.0
    total_uiqm = 0.0
    valid_count = 0

    # 遍历所有图片并计算UIQM
    for img_path in image_paths:
        try:
            # 读取图片
            img = cv2.imread(img_path)
            if img is None:
                print(f"无法读取图片: {img_path}")
                continue

            # 计算UIQM
            uiqm_value = getUIQM(img)
            total_uiqm += uiqm_value

            # calculate uciqe
            uciqe_value = uciqe(1,img)
            total_uciqe += uciqe_value

            valid_count += 1
            with open("UCIQE_UIQM.txt", "a") as f:
                f.write(f"image: {os.path.basename(img_path)}, UCIQE: {uciqe_value:.4f} UIQM: {uiqm_value:.4f}\n")
            print(f"处理图片: {os.path.basename(img_path)}, UCIQE: {uciqe_value:.4f} UIQM: {uiqm_value:.4f}")

        except Exception as e:
            print(f"处理图片 {img_path} 时出错: {str(e)}")

    # 计算平均值
    if valid_count == 0:
        print("没有有效图片可以计算")
        return 0.0,0.0, 0

    average_uiqm = total_uiqm / valid_count
    average_uciqe= total_uciqe / valid_count
    return average_uciqe,average_uiqm, valid_count


if __name__ == '__main__':
    # 替换为你的图片文件夹路径
    folder_path = "dir"

    # 计算平均值
    avg_uciqe,avg_uiqm, count = calculate_folder_uciqe_uiqm_average(folder_path)

    # 输出结果
    with open("UCIQE_UIQM.txt", "a") as f:
        f.write(f"folder {folder_path} number {count} average UIQM: {avg_uiqm:.4f},average UCIQE:{avg_uciqe:.4f}\n")
    print(f"\n文件夹 {folder_path} 中 {count} 张图片的平均UIQM值为: {avg_uiqm:.4f},平均UCIQE值为:{avg_uciqe:.4f}")
