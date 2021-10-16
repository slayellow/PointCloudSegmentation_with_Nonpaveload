import time
from multiprocessing import Process, Queue
import std_msgs.msg
from inference.pytorch_util import *
from inference.data_parser import *
import inference.config as cf
import rclpy
from rclpy.node import Node
from std_msgs import *
from sensor_msgs.msg import PointCloud, ChannelFloat32
from nav_msgs.msg import OccupancyGrid
from scipy.spatial import Delaunay
import multiprocessing
import threading

import torch.autograd
import torch.backends.cudnn as cudnn
from inference.SalsaNext import *
import yaml
from inference.KNN import KNN
import time

import torch

import numpy as np
from timeit import default_timer as timer

import os


class NonpaveloadSegmentor(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        # Load Parsing File
        self.model_info = yaml.safe_load(open("inference/inference/" + cf.paths["model_info"], 'r'))
        self.dataset_info = yaml.safe_load(open("inference/inference/" + cf.paths["dataset_info"], 'r'))
        self.color_map = self.dataset_info["color_map"]
        self.learning_map_inv = self.dataset_info["learning_map_inv"]

        # GPU Check
        gpu_check = is_gpu_avaliable()
        devices = torch.device("cuda") if gpu_check else torch.device("cpu")

        # Load Model
        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            self.model = salsanext(cf.NUM_CLASSES).to(devices)

        self.model.eval()
        pretrained_path = cf.paths['pretrained_path']
        if os.path.isfile(os.path.join(pretrained_path, self.model.get_name() + '.pth')):
            print("Pretrained Model Open : ", self.model.get_name() + ".pth")
            checkpoint = load_weight_file(os.path.join(pretrained_path, self.model.get_name() + '.pth'))
            load_weight_parameter(self.model, checkpoint['state_dict'])
        else:
            print("No Pretrained Model")
            return

        if is_gpu_avaliable() and get_gpu_device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True

        self.post = None
        if self.model_info["post"]["KNN"]["use"]:
            self.post = KNN(self.model_info["post"]["KNN"]["params"],
                       cf.NUM_CLASSES)

        self.vlp_subscriber = self.create_subscription(PointCloud, cf.VELODYNE_SUB, self.on_vlp, 100)
        print('Velodyne Subscriber Create Success : ', cf.VELODYNE_SUB)
        self._pts_mutex = threading.Lock()
        self._vlpdata = None

        self.vlp_publisher = self.create_publisher(PointCloud, cf.SEGMENTATION_POINTCLOUD, 100)
        print("Segmentation Point Cloud Publisher Create Success : ", cf.SEGMENTATION_POINTCLOUD)

        self.count = 1

        # thread = threading.Thread(target=self.on_thread)
        # thread.start()

    @staticmethod
    def mapping(label, mapdict):
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        return lut[label]

    def get_original(self, label):
        return self.mapping(label, self.learning_map_inv)

    def get_color(self, label):
        return self.mapping(label, self.color_map)

    def on_thread(self):
        while True:
            start = time.time()

            if self._vlpdata is not None:
                pointcloud_list = self._vlpdata.points
                pointcloud_list = [[pointcloud.x, pointcloud.y, pointcloud.z] for pointcloud in pointcloud_list
                                   if not (pointcloud.x == 0 and pointcloud.y == 0 and pointcloud.z == 0)]
                pointcloud_list = np.array(pointcloud_list)
                intensity_list = self._vlpdata.channels[0].values
                intensity_list = np.array(intensity_list) / np.array(255)

                proj, proj_x, proj_y, proj_range, unproj_range, num_points = get_inference_data(self.model_info, pointcloud_list, intensity_list)

                p_x = proj_x[0, :num_points]
                p_y = proj_y[0, :num_points]
                proj_range = proj_range[0, :num_points]
                unproj_range = unproj_range[0, :num_points]

                if is_gpu_avaliable():
                    proj = proj.cuda()
                    p_x = p_x.cuda()
                    p_y = p_y.cuda()
                    if self.post:
                        proj_range = proj_range.cuda()
                        unproj_range = unproj_range.cuda()

                proj_output = self.model(proj)
                proj_argmax = proj_output[0].argmax(dim=0)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                if self.post:
                    unproj_argmax = self.post(proj_range,
                                         unproj_range,
                                         proj_argmax,
                                         p_x,
                                         p_y)
                else:
                    unproj_argmax = proj_argmax[p_y, p_x]

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                pred_np = unproj_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)
                pred_np = self.get_original(pred_np)

                color_np = self.get_color(pred_np).astype(np.uint8).tolist()

                segment_pointcloud = PointCloud()

                label_result = ChannelFloat32(name="Intensity")
                label_result.values = pred_np.astype(np.float32).tolist()
                rgb_ch = ChannelFloat32(name='rgb')
                rgb_ch.values = [float((color[0]*0xFF*0xFF + color[1]*0xFF + color[2]) / 256) for color in color_np]

                segment_pointcloud.points = self._vlpdata.points
                segment_pointcloud.channels = [rgb_ch, label_result]
                segment_pointcloud.header.frame_id = cf.FRAME_ID

                self.vlp_publisher.publish(segment_pointcloud)

                print("Algorithm Processing Time : ", time.time()-start)

    def on_vlp(self, msg):
        self._vlpdata = msg
        end = time.time()

        pointcloud_list = []
        for idx in range(len(msg.points)):
            pointcloud = msg.points[idx]
            intensity = msg.channels[0].values[idx]
            if pointcloud.x == 0 and pointcloud.y ==0 and pointcloud.z ==0:
                continue
            else:
                pointcloud_list.append([pointcloud.x, pointcloud.y, pointcloud.z, intensity/255])
        pointcloud_list = np.array(pointcloud_list, dtype=np.float32)

        path = os.path.join(cf.paths["log_save_path"], str(self.count).zfill(6) + ".bin")
        pointcloud_list.tofile(path)
        print("Index : " + str(self.count) + ", Loading Time : " + str(time.time() - end) + ", Path : " + path)
        self.count = self.count + 1


def main(args=None):
    rclpy.init(args=args)
    inference = NonpaveloadSegmentor('nonpaveload_segmentor')
    rclpy.spin(inference)
    inference.multi_process.join()
    inference.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
