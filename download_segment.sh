!git clone https://github.com/VainF/DeepLabV3Plus-Pytorch.git
%cd DeepLabV3Plus-Pytorch

!pip install -r requirements.txt

# single image
!python predict.py --input image.jpg  --dataset cityscapes --model deeplabv3plus_resnet --ckpt checkpoints/best_deeplabv3plus_resnet_cityscapes_os16.pth --save_val_results_to test_results

# image folder
#!python predict.py --input datasets/data/cityscapes/leftImg8bit/train/bremen  --dataset cityscapes --model deeplabv3plus_mobilenet --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth --save_val_results_to test_results