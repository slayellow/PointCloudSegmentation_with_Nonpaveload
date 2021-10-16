import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from DataManagement.LaserScan import LaserScan, SemLaserScan
import UtilityManagement.config as cf
import os
import matplotlib.pyplot as plt
import cv2
import struct
import yaml



# Log File Parsing
def get_pointcloud(data):
    xyz_list = []
    intensity_list = []

    for idx in range(230400):
        x, y, z, layer, intensity = struct.unpack('hhhBB', data[idx*8:idx*8+8])
        xyz_list.append([x * 0.01 ,y * 0.01 ,z * 0.01])
        intensity_list.append([intensity])
    return xyz_list, intensity_list


# Get Data required inference
def get_inference_data(model_info, xyz, intensity):

    sensor = model_info["dataset"]["sensor"]
    sensor_img_H = sensor["img_prop"]["height"]
    sensor_img_W = sensor["img_prop"]["width"]
    sensor_fov_up = sensor["fov_up"]
    sensor_fov_down = sensor["fov_down"]
    sensor_img_means = torch.tensor(sensor["img_means"], dtype=torch.float)
    sensor_img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)
    max_points = 250000                 # VLP128 : 230400, OS1-128 : 231600

    proj_range = np.full((sensor_img_H, sensor_img_W), -1, dtype=np.float32)
    proj_xyz = np.full((sensor_img_H, sensor_img_W, 3), -1, dtype=np.float32)
    proj_remission = np.full((sensor_img_H, sensor_img_W), -1, dtype=np.float32)
    proj_idx = np.full((sensor_img_H, sensor_img_W), -1, dtype=np.int32)

    # laser parameters
    fov_up = sensor_fov_up/ 180.0 * np.pi           # field of view up in rad
    fov_down = sensor_fov_down / 180.0 * np.pi      # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)               # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(xyz, 2, axis=1)

    # get scan components
    scan_x = xyz[:, 0]
    scan_y = xyz[:, 1]
    scan_z = xyz[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= sensor_img_W          # in [0.0, W]
    proj_y *= sensor_img_H          # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(sensor_img_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
    proj_x_original = np.copy(proj_x)                   # Original Order Point

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(sensor_img_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
    proj_y_original = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    unprojection_range_original = np.copy(depth)

    # order in decreasing depth         # 거리에 대해 내림차순 정
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = xyz[order]
    remission = intensity[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    proj_range[proj_y, proj_x] = depth
    proj_xyz[proj_y, proj_x] = points
    proj_remission[proj_y, proj_x] = remission
    proj_idx[proj_y, proj_x] = indices
    proj_mask = (proj_idx > 0).astype(np.int32)

    num_points = xyz.shape[0]

    unproj_range = torch.full([max_points], -1.0, dtype=torch.float)
    unproj_range[:num_points] = torch.from_numpy(unprojection_range_original)

    proj_range = torch.from_numpy(proj_range).clone()
    proj_xyz = torch.from_numpy(proj_xyz).clone()
    proj_remission = torch.from_numpy(proj_remission).clone()
    proj_mask = torch.from_numpy(proj_mask)


    proj_x = torch.full([max_points], -1, dtype=torch.long)
    proj_x[:num_points] = torch.from_numpy(proj_x_original)
    proj_y = torch.full([max_points], -1, dtype=torch.long)
    proj_y[:num_points] = torch.from_numpy(proj_y_original)
    proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()])
    proj = (proj - sensor_img_means[:, None, None]) / sensor_img_stds[:, None, None]
    proj = proj * proj_mask.float()


    # proj : Original Point Cloud -> Projection -> index > 0 Projection Point Cloud
    # proj_x : Original Point Cloud -> Projection -> X Coordinate ( Sorting X )
    # proj_y : Original Point Cloud -> Projection -> Y Coordinate ( Sorting X )
    # proj_range : Original Point Cloud -> Projection -> Order Depth
    # unproj_range : Original Point Cloud -> Depth ( Sorting X )
    return proj, proj_x, proj_y, proj_range, unproj_range,  num_points


