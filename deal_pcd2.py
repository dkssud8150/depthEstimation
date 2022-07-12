import open3d
import numpy as np
import yaml
import cv2
import os,sys
import matplotlib.pyplot as plt
from PIL import Image


from recon import pltImg

def computemetric(gt, pred, metric="rmse", eps = 1e-08):
    threshold = np.maximum((gt / (pred + eps)), (pred / (gt + eps)))
    if metric == 'd1':
        return (threshold < 1.25).mean()
    elif metric == 'd2':
        return (threshold < 1.25 ** 2).mean()
    elif metric == 'd3':
        return (threshold < 1.25 ** 3).mean()
    elif metric == 'rmse':
        return np.sqrt(((gt - pred) ** 2).mean())
    elif metric == 'rmselog':
        if gt == 0 or pred == 0: return 0
        return np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    elif metric == 'abs_rel':
        return np.mean(np.abs(gt - pred) / gt)
    elif metric == 'sq_rel':
        return np.mean(((gt - pred) ** 2) / gt)
    else:
        raise ValueError


def draw_pcd(pcd_array):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pcd_array)
    open3d.visualization.draw_geometries([pcd])
    
def read_pld(filename):
    pcd = open3d.io.read_point_cloud(filename, format="pcd")
    pcd_array = np.array(pcd.points)
    
    pcd_array = pcd_array[ ~np.isnan(pcd_array).any(axis=1),:]  # Nan 행 제거
    
    return pcd_array

def parse_camera(f_path):
    with open(f_path) as f:
        calib_cam = {}
        calib_info = yaml.load(f, Loader=yaml.FullLoader)["camera"]["front"]
        calib_cam["P"] = calib_info["P"] 
        calib_cam["R"] = calib_info["R"]  
        calib_cam["t"] = calib_info["T"]  
        calib_cam["size"] = calib_info["size"]  
    return calib_cam

def parse_lidar(f_path):
    with open(f_path) as f:
        calib_lidar = {}
        calib_info = yaml.load(f, Loader=yaml.FullLoader)["lidar"]["rs80"]
        calib_lidar["R"] = calib_info["R"]
        calib_lidar["t"] = calib_info["T"]
    return calib_lidar


def in_image(point, size):
    row = np.bitwise_and(0 <= point[0], point[0] < size["width"])
    col = np.bitwise_and(0 <= point[1], point[1] < size["height"])
    return np.bitwise_and(row, col)

def lidar_to_camera(calib_cam, calib_lidar, points): 
    proj = []
    for p in points:
        proj_point = project_point(p,calib_cam, calib_lidar)
        if in_image(proj_point, calib_cam["size"]) and 0 <= proj_point[2]:
            proj.append(proj_point)
    return np.array(proj)


def project_point(point, calib_cam, calib_lidar):
    ## [x,y,z,1]
    lidar = np.append(point, [1], axis=0)
    lidar = np.transpose(lidar)
    # [R|t]X[x,y,z,1]
    matrix_lidar = np.concatenate([calib_lidar["R"], calib_lidar["t"]], axis=1)
    matrix_lidar = np.matmul(matrix_lidar, lidar)
    matrix_lidar = np.append(matrix_lidar, [1], axis=0) 

    P = np.array(calib_cam["P"]) 
    matrix_cam= np.concatenate([calib_cam["R"], calib_cam["t"]], axis=1)
    matrix_cam = np.concatenate([matrix_cam, [[0, 0, 0, 1]]], axis=0)  
    matrix_cam = np.matmul(P, matrix_cam) 
    matrix_cam = np.matmul(matrix_cam, matrix_lidar)  # (3,1) s*[x,y,1]
    
    depth = matrix_cam[-1]
    
    # x,y 정보
    matrix_cam[:-1] = matrix_cam[:-1] / depth 
    matrix_cam[:-1] = np.array(list(map(int, matrix_cam[:-1])))
    
    return matrix_cam

def save_depth_gt(depth_gt, calib_cam):
    img = np.zeros((calib_cam["size"]["height"], calib_cam["size"]["width"]), dtype=np.float32)
    
    for x, y, d in depth_gt:
        if img[int(y)][int(x)] == 0:
            img[int(y)][int(x)] = d
        else:
            if img[int(y)][int(x)] > d:
                img[int(y)][int(x)] = d
                
    os.makedirs("data/img/acelab/gangnam1/depth_gt/", exist_ok=True)
    cv2.imwrite("data/img/acelab/gangnam1/depth_gt/00000.png", img)



def projection(img_path, depth_path, depth_gt):
    img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    depth_map = cv2.imread(depth_gt, cv2.IMREAD_ANYDEPTH)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    depth_map = cv2.resize(depth_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    for v in range(img.shape[0]):
        for u in range(img.shape[1]):
            pred = depth[v,u]
            gt = depth_map[v,u]
            
            if gt != 0: cv2.circle(img, (u,v), 1, (255,0,0), thickness=1)

    pltImg([img, depth, depth_map])

    plt.imshow(img)
    plt.show()


def metrics(img_path, depth_path, depth_map):
    img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    depth_map = cv2.imread(depth_map, cv2.IMREAD_ANYDEPTH)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    depth_map = cv2.resize(depth_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    rmse, rmselog = 0,0
    preds, gts = list(), list()
    for v in range(img.shape[0]):
        for u in range(img.shape[1]):
            rgb = img[v,u]
            pred = depth[v,u]
            gt = depth_map[v,u]

            # if gt == 0: continue
            # else : print(pred, gt)

            rmse += computemetric(gt, pred, "rmse")
            rmselog += computemetric(gt, pred, "rmselog")

            preds.append(pred)
            gts.append(gt)

    plt.subplot(1,2,1)
    plt.hist(preds, bins=255)
    plt.subplot(1,2,2)
    plt.hist(gts ,bins= 255)
    plt.show()

    print(f"pred max : {np.nanmax(depth)}\t min : {np.nanmin(depth)}\ngt max : {np.nanmax(depth_map)}\t min : {np.nanmin(depth_map)}")

    print(depth[:5], "\n", depth_map[:5])

    rmse = rmse.mean()
    rmselog = rmselog.mean()

    print(rmse, rmselog)

    # pltImg([img,depth,depth_map])




if __name__ == "__main__":
    
    calib_file = "data/calibration/acelab/calibration.yaml"
    pcd_file = "data/img/acelab/gangnam1/gangnam1_pcd/1656481329.238302708.pcd"

    img_path = "data/img/acelab/gangnam1/origin/00000.png"
    depth_path = "data/img/acelab/gangnam1/depth/Adabins/00000.png"
    depth_map = "data/img/acelab/gangnam1/depth_gt/00000.png"

    calib_cam = parse_camera(calib_file)
    calib_lidar = parse_lidar(calib_file)
    
    points = read_pld(pcd_file)
    
    depth_gt = lidar_to_camera(calib_cam, calib_lidar, points)

    # save_depth_gt(depth_gt, calib_cam)
    # projection(img_path, depth_path, depth_map)

    metrics(img_path, depth_path, depth_map)
