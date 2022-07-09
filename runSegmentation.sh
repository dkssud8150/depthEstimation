# run segmentation for kitti dataset using single gpu
# python test.py --dataset <datasetname> --gpu <gpu number> --save_dir <save dir>

datasetname=$1
GPUS=$2

python segment.py --dataset datasetname --gpu GPUS
