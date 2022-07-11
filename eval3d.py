'''
input : calibration file, pcd file, depth image, RGB image
1. convert image and depth to 3d point cloud 
2. compare x,y,z for image data and pcd

output : metrics
'''

from glob import glob
from shutil import SpecialFileError
import cv2
import numpy as np
from PIL import Image
import open3d
import pickle
import os

from recon import parsing, checkintrinsic, pltImg, projection


def convertpcdtoImg(pcdPaths, calibrationFile, datasetName, originPath):
    for pcdPath in pcdPaths:
        pcd = open3d.io_read_point_cloud(pcdPath)
        pcdNumpy = np.asarray(pcd.points, dtype=np.float32)
    
        originImg = cv2.imread(originPath)


def convertImgtopcd(originPath, depthPath, calibrationFile, datasetName):
    originImg = cv2.imread(originPath)
    depthImg = np.asarray(Image.open(depthPath).convert('I'), dtype="uint16")

    K,R,T,D = parsing(calibrationFile, datasetName)
    
    
    K_inv = np.linalg.inv(K)
    R_inv = np.linalg.inv(R)

    scaleFactor = 1

    worldpoints = list()
    colors = list()

    for v in range(originImg.shape[0]):
        for u in range(originImg.shape[1]):
            color = originImg[v][u]
            depth = depthImg[v][u]

            uv = np.array([[u,v,1]], dtype=np.float64)
            uv_t = uv.T
            suv = scaleFactor*uv_t
            xyz = K_inv.dot(suv)
            xyz = xyz - T
            XYZ = R_inv.dot(xyz)

            XYZ[0] = XYZ[0]
            XYZ[1] = -XYZ[1]
            XYZ[2] = -XYZ[2] + (depth / 1000)

            worldpoints.append(XYZ.T)
            colors.append(color)


    # os.makedirs("data/txt/", exist_ok=True)
    # with open("data/txt/" + originPath.split('\\')[-1].split('.')[0] + ".txt", 'wb') as f:
    #     pickle.dump(worldpoints, f)

    return np.array(worldpoints, np.float64).reshape(-1,3), np.array(colors, np.float64)


def compareImgandpcd(datasetName, worldpoints, colors, pcdPath):
    # for pcdPath in pcdPaths:
    pcd = open3d.io.read_point_cloud(pcdPath)
    pcdNumpy = np.asarray(pcd.points, dtype=np.float32)


    print(pcdNumpy.shape)
    print(worldpoints.shape)

    pcdall = np.concatenate([worldpoints, pcdNumpy])

    print(pcdall.shape)

    lidarcolors = np.array([[[np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255)]] * pcdNumpy.shape[0]], dtype=np.float64).reshape(-1,3)
    print(lidarcolors[:5])
    print(lidarcolors.shape)
    print(colors.shape)
    colorsall = np.concatenate([colors, lidarcolors])

    # rmse = np.sqrt(((gt - pred) ** 2).mean())

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pcdall)
    pcd.colors = open3d.utility.Vector3dVector(colorsall / 255.)

    ## visualization
    # open3d.visualization.draw_geometries([pcd],
    #                               window_name='open3d',
    #                               zoom=0.3412,
    #                               front=[0.4257, -0.2125, -0.8795],
    #                               lookat=[2.6172, 2.0475, 1.532],
    #                               up=[-0.0694, -0.9768, 0.2024])

    # save file
    save_path = "ply/" + datasetName + "/" + originPath.split('\\')[-1].split('.')[0] + ".ply"
    open3d.io.write_point_cloud(save_path, pcd)




if __name__ == "__main__":
    pcdPaths = sorted(glob("gangnam_pcd/*"), key=lambda x : x.split('/')[-1])
    pcdPath = pcdPaths[0]
    
    datasetName = 'acelab'
    img_paths = sorted(glob("data/img/"+datasetName+"/gangnam2/origin/*"), key = lambda x : x.split('/')[-1])[:1]

    for i,img_path in enumerate(img_paths):
        print(f"\n\n{i} th converting.. {img_path.split('/')[-1]}")
        originPath = img_path
        depthPath = img_path.replace('origin', 'depth')
        calibrationFile = glob("config/" + datasetName + "/*")[0]


        worldpoints, colors = convertImgtopcd(originPath, depthPath, calibrationFile, datasetName)
        compareImgandpcd(datasetName, worldpoints, colors, pcdPath)

