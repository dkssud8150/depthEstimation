# image2lidar

convert image to 3d point cloud using depth image and RGB image

실행 과정

1. open3D 패키지를 설치한다. `!pip install opencv-python open3d`
2. 변환하고자 하는 데이터셋의 원본 이미지와 depth map 이미지를 `data/img/<datasetname>/origin` and `depth`
3. `runReconstruction.sh` 실행

<br>

# segmentation

미리 학습된 deeplabv3-resnet50 weight 파일을 가져와서 추론한다.

실행 과정

1. 변환하고자 하는 이미지를 가져온다. 디렉토리여도 되고, 이미지여도 된다.
2. `runSegmentation.sh`를 자신이 원하는 dataset이름과 이미지 디렉토리를 지정하여 실행한다.
    - e.g. `sh runSegmentation.sh /img/kitti/origin/ 1`
