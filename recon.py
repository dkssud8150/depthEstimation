'''
reconstruction image using pcd and depth map
- input : pcd, image, depth map, calibration file
1. convert depth map to 3d points by calibration for each pixel
    - reference : https://github.com/mcordts/cityscapesScripts/blob/master/docs/csCalibration.pdf
                : https://github.com/mcordts/cityscapesScripts/blob/master/docs/Box3DImageTransform.ipynb
2. convert image data to point cloud using the depth value at each pixel
3. calculate loss between this image point cloud and pcd
4. remove noise 
    - ground plane remove in lidar data
    - RANSAC
- output : ply file
'''

from PIL import Image
import cv2
import open3d
import numpy as np
import yaml
import math
import matplotlib.pyplot as plt
from tqdm.notebook import trange
import os
import sys
from glob import glob

def checkintrinsic(K,R,T,D,originImg):
    ######## step 3. undistort
    h,w = originImg.shape[:2]

    NK , roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), alpha=1, newImgSize=(w,h))
    dst = cv2.undistort(originImg, K, D, None, NK)

    # pltImg(originImg, dst)


def pltImg(img1, img2):
    plt.figure(figsize=(15,9))
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.show()


def parsing(calibrationFile, type='kitti'):
    if type != 'kitti':
        with open(calibrationFile, 'r') as c:
            calib = yaml.load(c, Loader=yaml.FullLoader)
    elif type == 'kitti':
        calib = dict()
        with open(calibrationFile, 'r') as c:
            lines = c.readlines()
            for line in lines:
                line = line.strip()
                keys, values = line.split(':', 1)
                calib[keys.strip()] = list(values.strip().split(' '))

    if type=='cityscapes':
        fx, fy, cx, cy = calib['intrinsic']['fx'], calib['fy'], calib['u0'], calib['v0']
        K = [
                fx, 0,  cx,
                0,  fy, cy,
                0,  0,  1
            ]

        RT = calib['extrinsic']

        baseline, pitch, roll, yaw, x, y, z = RT['baseline'], RT['pitch'], RT['roll'], RT['yaw'], RT['x'],RT['y'],RT['z']

        cy = math.cos(yaw)
        sy = math.sin(yaw)
        cp = math.cos(pitch)
        sp = math.sin(pitch)
        cr = math.cos(roll)
        sr = math.sin(roll)

        R = [
            cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr,
            sy * cp,  sy * sp * sr + cy * cr,    sy*sp*cr - cy * sr, 
            -sp, cp * sr, cp * cr
            ].transpose()

        T = R * [x,y,z]

        return K, R, T
    elif type=='acelab':
        K = calib['camera']['front']['K']
        R = calib['camera']['front']['R']
        T = calib['camera']['front']['T']
        D = calib['camera']['front']['D']

        lidar_info = calib['lidar']
        

        return np.array(K, dtype=np.float32),np.array(R, dtype=np.float32),np.array(T, dtype=np.float32),np.array(D, dtype=np.float32)

    elif type=='kitti':
        K = np.array(calib['K_03'], dtype=np.float64).reshape(3,3)
        R = np.array(calib['R_03'], dtype=np.float64).reshape(3,3)
        T = np.array(calib['T_03'], dtype=np.float64).reshape(3,1)
        D = np.array(calib['D_03'], dtype=np.float64).reshape(1,5)
        P = np.array(calib['P_rect_03'], dtype=np.float64).reshape(3,4)

        return K,R,T,D



def tools(originPath, depthPath, calibrationFile):
    # Can you check the color image is: 8bit 1channel or 8bit 3channel, 
    # and depth image is: 16bit 1channel image? By default Open3D regard value 1000 in depth image as 1000mm, or 1m.
    colors = np.asarray(Image.open(originPath), dtype='uint8') 
    depth = np.asarray(Image.open(depthPath).convert('I'), dtype="uint16") * 255

    colors = open3d.geometry.Image(colors)
    depth = open3d.geometry.Image(depth) 

    rgbd = open3d.geometry.RGBDImage.create_from_color_and_depth(colors, depth, depth_trunc= 1000)
    print(np.asarray(rgbd))

    K,R,T,D = parsing(calibrationFile, type='kitti')
    K = open3d.camera.PinholeCameraIntrinsic(1245,375,K[0][0],K[1][1],K[0][2],K[1][2])

    # pltImg(rgbd.color, rgbd.depth)

    pcd = open3d.geometry.PointCloud.create_from_rgbd_image(rgbd, K)
    pointcloud = open3d.geometry.PointCloud.create_from_depth_image(depth, K)

    print(np.asarray(pcd.colors))
    print("\n")
    print(np.asarray(pointcloud.points))


    pcd.transform([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
    
    # open3d.visualization.draw_geometries([pcd])
    # open3d.visualization.draw_geometries([pointcloud])

    open3d.io.write_point_cloud("a.ply", pointcloud)



def projection(K, R, T, originImg, depthImg, scaleFactor = 256, type=2): # in depthformer, depth scale = 256, z = disparity 
    #### convert original image to 3d points and plus depth value each pixel
    worldpoints = list()
    colors = list()
    if type==0: # 한 점으로 표시됨. scale을 키워도 한점으로 되어 있다.
        fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
        
        # http://www.open3d.org/docs/0.7.0/python_api/open3d.geometry.create_point_cloud_from_depth_image.html
        # https://goodgodgd.github.io/ian-flow/archivers/kros-ss-2019-note3
        for v in trange(originImg.shape[0]):
            for u in range(originImg.shape[1]):
                color = originImg[v][u]
                depth = depthImg[v][u]     # typically millimeter scale
                Z = depth / scaleFactor
                X = (u - cx) / fx * Z   # 좌우      # simplest matrix mul without extrinsic matrix
                Y = (v - cy) / fy * Z   # 위아래

                worldpoints.append([X, Y, Z]) # -0.02105, -0.006798, 0.02734
                colors.append(color)

        # 이렇게 계산한 points가 world가 아니라 camera인거 아닌가, 아래 식이 맞는듯
    
        ''' 1과 2는 동일하게 2d 사진으로 표현된다. 1번이 조금 더 작고, 2번이 딱 2배다. 더 간결한 2번을 사용해보고자 함. '''
    # [x y z 1] = z [[1/fx 0 0 0],[0 1/fy 0 0],[0 0 1 0],[0 0 0 1]] [u v 1 1/z]
    ## 2dpoint * inv(K.dot(RT)) = 3dpoint 
    ## get world coordinate without depth value
    elif type==1:
        arr1 = np.zeros((1,3), dtype=np.float64)
        K34 = np.concatenate([np.array(K), arr1], axis=0)

        arr2 = np.array([[0,0,0,1]],dtype=np.float64).T
        K = np.concatenate([K34,arr2], axis=1)

        RT = np.concatenate([R,T], axis=1)
        arr3 = np.array([[0,0,0,1]],dtype=np.float64)
        RT = np.concatenate([RT,arr3], axis=0)
        
        mulmtx = K.dot(RT)
        projectionMatrix = np.linalg.inv(mulmtx)

        for v in trange(originImg.shape[0]):
            for u in range(originImg.shape[1]):
                color = originImg[v][u]
                depth = depthImg[v][u]

                z = depth / scaleFactor
                worldpoint = z * projectionMatrix.dot(np.array([u,v,1,1/z]))

                worldpoints.append(worldpoint[:3]) # 0.4528, -0004592, 0.02734
                colors.append(color)


    else:
        ## https://www.fdxlabs.com/calculate-x-y-z-real-world-coordinates-from-a-single-camera-using-opencv/
        # 역행렬을 수행하려면 정사각행렬이어야 한다. 따라서 4x4 행렬로 만들어줘야 함
        # [u v 1] = 1/z K [R T] [x y z 1]
        K_inv = np.linalg.inv(K)
        R_inv = np.linalg.inv(R)

        
        for v in range(originImg.shape[0]):
            for u in range(originImg.shape[1]):
                color = originImg[v][u]
                depth = depthImg[v][u]

                uv = np.array([[u,v,1]], dtype=np.float64)
                uv_t = uv.T
                suv = scaleFactor*uv_t
                xyz = K_inv.dot(suv) # camera coordinate
                xyz = xyz - T
                XYZ = R_inv.dot(xyz)

                XYZ[0] = XYZ[0] * 1          # scale을 위해 x를 곱함 , 좌우 간격
                XYZ[1] = XYZ[1] * 2          # scale을 위해 y를 곱함 , 위아래 간격
                XYZ[2] = XYZ[2] + depth / 50 # scale을 위해 50을 나눔 , depth 깊이

                
                worldpoints.append(XYZ.T)
                colors.append(color)
        
    return np.array(worldpoints, np.float64).reshape(-1,3), np.array(colors, np.float64)
        

def reconstruction(pcdFile, originPath, depthPath, calibrationFile, datasetName):
    # 0. prepare info
    TYPE = 2
    os.makedirs("ply/" + datasetName, exist_ok=True)
    save_path = "ply/" + datasetName + "/" + originPath.split('/')[-1].split('.')[0] + ".ply"

    pcd = open3d.io.read_point_cloud(pcdFile)
    pcdNumpy = np.asarray(pcd.points)
    # print(pcd)
    # print(type(pcd))


    originImg = cv2.imread(originPath)
    depthImg = np.asarray(Image.open(depthPath).convert('I'), dtype="uint16")

    # pltImg(originImg, depthImg)

    K,R,T,D = parsing(calibrationFile, datasetName)

    checkintrinsic(K,R,T,D, originImg)


    # 1. convert depth map to 3d points by calibration for each pixel
    points, colors = projection(K, R, T, originImg, depthImg, TYPE)


    print(points[:5], "\n")

    
    # convert numpy to Open3D
    print("converting finish")

    pcd = open3d.geometry.PointCloud()

    pcd.points = open3d.utility.Vector3dVector(points)
    pcd.colors = open3d.utility.Vector3dVector(colors / 255.)

    ## visualization
    # open3d.visualization.draw_geometries([pcd],
    #                               window_name='open3d',
    #                               zoom=0.3412,
    #                               front=[0.4257, -0.2125, -0.8795],
    #                               lookat=[2.6172, 2.0475, 1.532],
    #                               up=[-0.0694, -0.9768, 0.2024])

    # save file
    open3d.io.write_point_cloud(save_path, pcd)


    # write_nolib(pointsNumpy, points, colors, save_path)


if __name__ == "__main__":
    pcd = "gangnam_pcd/1656481494.537210464.pcd"
    datasetName = 'kitti'    
    img_paths = sorted(glob("img/"+datasetName+"/origin/*"), key = lambda x : x.split('/')[-1])

    for i,img_path in enumerate(img_paths):
        print(f"\n\n{i} th converting.. {img_path.split('/')[-1]}")
        originPath = img_path
        depthPath = img_path.replace('origin', 'depth')
        calibrationFile = glob("config/" + datasetName + "/*")[0]

        # tools(originPath, depthPath, calibrationFile)
        reconstruction(pcd, originPath, depthPath, calibrationFile, datasetName)