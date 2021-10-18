import numpy as np


# 데이터셋 관련 경로
paths = dict(
        dataset_info = "semantic-kitti.yaml",
        model_info = "salsanext.yml",
        pretrained_path = "/home/HONG/PretrainedParameter/PointCloudSegmentation",
        save_path = "/home/HONG/Semantic-KITTI/Result",
        log_path = "/home/HONG/Semantic-KITTI/2020-11-18_09-53-12.velodyne128xyz",
        log_save_path = "/home/HONG/Semantic-KITTI/Changwon_Nonpaveload/dataset/sequences/95/velodyne",
        changwon_save_path = "/home/HONG/Semantic-KITTI/Changwon_Nonpaveload/dataset/sequences/95/label"
)

FRAME_ID = "default"
VELODYNE_SUB = "TruckMaker/Sensor/VelodyneFCPointCloud"
OS1_128_SUB = "TruckMaker/Sensor/Ouster128PointCloud"
OS1_32_SUB = "TruckMaker/Sensor/Ouster32PointCloud"
SEGMENTATION_POINTCLOUD = "Segmentation_PointCloud"
PROBMAP_PUB = "Probability_Map"

SE_VECTOR = [-90.0, 0.0, 0.0, 0.0, -0.622, -0.091]

INTRINSIC_PARAMETER = np.array(([
    [320, 0, 320],
    [0, 240, 240],
    [0, 0, 1]
]))

IMG_WIDTH = 480
IMG_HEIGHT = 640
NUM_CLASSES = 20
MAP_SIZE = 300