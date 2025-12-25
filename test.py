import argparse
import os
import torch
import warnings
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.myutils.test_dataloader import UIEBD_Dataset
from src.UDHE_arch import LightweightRestorationNet as UDHE
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_type', type=str, default='dirname')
parser.add_argument('--dataset', type=str, default='your/data/degraded/image/input', help='path of UIED')
parser.add_argument('--savepath', type=str, default='enhanced/image/save/dir', help='path of output image')
parser.add_argument('--model_path', type=str, default='your/model/weight', help='path of model checkpoint')
opt = parser.parse_args()

val_set = UIEBD_Dataset(opt.dataset,256)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)


netG_1 = UDHE()#(dim=32)#.cuda()

if __name__ == '__main__':   

    ssims = []
    psnrs = []
    rmses = []
    
    g1ckpt1 = opt.model_path
    ckpt = torch.load(g1ckpt1,map_location=torch.device('cpu'))

    model_state_dict = ckpt#["state_dict"]

    # 过滤掉 "loss_per." 相关的参数
    filtered_state_dict = {}
    for k, v in model_state_dict.items():
        # 跳过损失模块的参数
        if "loss" in k:
            continue
        if "vgg" in k:
            continue
       
        if k.startswith("module."):
            filtered_k = k[7:]  # 去除 "module." 前缀
        else:
            filtered_k = k  # 无前缀则直接使用
        filtered_state_dict[filtered_k] = v

    #加载过滤后的参数
    netG_1.load_state_dict(filtered_state_dict,strict=True)

    netG_1.eval()
   
    savepath_dataset = os.path.join(opt.savepath,opt.dataset_type)
    if not os.path.exists(savepath_dataset):
        os.makedirs(savepath_dataset)
    loop = tqdm(enumerate(val_loader),total=len(val_loader))
    name = 0
    for batch_idx, batch in loop:
        clean_name_list, raw = batch
        with torch.no_grad():
            # 模型推理shape: [batch_size, C, H, W]）
            raw = raw  # 若使用GPU可改为 raw = raw.cuda()
            #print(raw.shape)
            enhancement_img = netG_1(raw)  # shape: [batch_size, C, H, W]

        # 遍历批量中的每张图片，单独保存
        for i in range(enhancement_img.shape[0]):
            # 获取当前图片的文件名
            img_name = clean_name_list[i] 

         
            save_path = os.path.join(savepath_dataset, img_name)

            # 保存单张图片
            save_image(enhancement_img[i], save_path, format='png', normalize=False)

        # 更新进度条信息
        loop.set_description(f"Processing batch {batch_idx + 1}/{len(val_loader)}")

                
