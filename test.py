# https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/
# https://colab.research.google.com/drive/1dWg0nx7KEYGSH05heY2_z5hosHBK3EbP#scrollTo=oGLyq3wY1CeO

import torch
import torch.nn
import torch.utils
from torch.utils.data import dataset
import torchvision
from torchvision import transforms
import torchvision.models

from tqdm.notebook import tqdm, trange

import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import argparse
import os
import cv2

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='cityscapes', choices=['voc', 'cityscapes', 'kitti', 'custom'], help='what is dataset name')
    parser.add_argument("--ckpt", type=str, default=None, help="where is your checkpoints")
    parser.add_argument("--gpu", type=str, default=0, help="if you want to use gpu, you set gpu id")
    parser.add_argument("--input", type=str, default='img/kitti/origin/',help="set you want to predict file or dir")
    parser.add_argument("--save_dir", type=str, default='result')

    return parser

def pltImg(imgs):
    length = len(imgs)
    plt.figure(figsize=(length*4, length))
    for i,img in enumerate(imgs):
        plt.subplot(1,len(imgs),i+1)
        plt.imshow(img)
    plt.show()


def main():
    args = parsing().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)

    if args.ckpt is not None and os.path.isfile(args.ckpt):
        checkpoint = torch.load(args.ckpt, map_location=device)
    
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
    ])

    os.makedirs(args.save_dir, exist_ok=True)

    print("segmentation start.. ")

    if os.path.isdir(args.input):
        print("we use directory")
        img_paths = sorted(glob(args.input + args.dataset + "/origin/*"), key = lambda x : x.split('/')[-1])
        if len(img_paths) == 0: 
            print("do not exist img in directory so we use default image, kitti")
            img_paths = sorted(glob("img/kitti/origin/*"), key = lambda x : x.split('/')[-1])
        
        with torch.no_grad():
            for img_path in img_paths:
                model = model.eval()
                img = Image.open(img_path).convert('RGB')
                imgd = img.copy()
                img_size = img.size
                img = transform(img).unsqueeze(0)
                img = img.to(device)

                pred = model(img)['out'][0].argmax(0).byte().cpu().numpy()

                # create color
                palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
                colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
                colors = (colors % 255).numpy().astype("uint8")
                
                pred = Image.fromarray(pred).resize(img_size)
                pred.putpalette(colors)
                
                
                # https://github.com/nkmk/python-snippets/blob/a613092203697247a5999eab69b222c2c2a4a497/notebook/pillow_composite.py#L11-L13
                saveImg = Image.composite(pred, imgd, pred.convert('L')) 

                saveImg.save(args.save_dir + "/" + img_path.split('/')[-1])
                print("save img file , ", img_path.split('/')[-1])

        pltImg([imgd,pred,saveImg])

    elif os.path.isfile(args.input):
        img_path = args.input

        with torch.no_grad():
            model = model.eval()
            img = Image.open(img_path).convert('RGB')
            imgd = img.copy()
            img_size = img.size
            img = transform(img).unsqueeze(0)
            img = img.to(device)

            pred = model(img)['out'][0].argmax(0).byte().cpu().numpy()

            # create color
            palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
            colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
            colors = (colors % 255).numpy().astype("uint8")
            
            pred = Image.fromarray(pred).resize(img_size)
            pred.putpalette(colors)

            saveImg = Image.composite(pred, imgd, pred.convert('L'))

            pltImg([imgd,pred,saveImg])

            pred.save(args.save_dir + "/" + img_path.split('/')[-1])
            print("save img file , ", img_path.split('/')[-1])
            

    else:
        raise Exception("디렉토리가 잘못되었습니다.")
    




if __name__ == "__main__":
    main()