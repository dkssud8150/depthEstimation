# run reconstruction using depth image and RGB image

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html

git clone https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox.git
cd Monocular-Depth-Estimation-Toolbox
pip install -e .

pip install future tensorboard

python ./tools/test.py ./configs/depthformer/depthformer_swinl_22k_w7_kitti.py ./checkpoints/depthformer_swinl_22k_kitti.pth --show-dir depthformer_swinl_22k_w7_kitti_result