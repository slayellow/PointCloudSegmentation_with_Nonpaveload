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

import yaml

EXTENSIONS_SCAN = ['.bin']      # Dataset File Extension
EXTENSIONS_LABEL = ['.label']   # Label File Extension


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class SemanticKitti(Dataset):
    def __init__(self, model_info, dataset_info, transform=True, mode=0):
        self.model_info = model_info
        self.dataset_info = dataset_info
        self.transform = transforms
        self.mode = mode        # 0 : Train, 1 : Validation, 2 : Test
        self.gt = False

        self.root = os.path.join(self.model_info["dataset"]["path"], "sequences")

        if self.mode == 0:
            self.sequences = self.dataset_info["split"]["train"]
            self.gt = True
        elif self.mode == 1:
            self.sequences = self.dataset_info["split"]["valid"]
            self.gt = True
        elif self.mode == 2:
            self.sequences = self.dataset_info["split"]["test"]
            self.gt = False
        else:
            print("No Suitable Mode!")
            self.sequences = None

        self.labels = self.dataset_info["labels"]
        self.color_map = self.dataset_info["color_map"]
        self.learning_map = self.dataset_info["learning_map"]
        self.learning_map_inv = self.dataset_info["learning_map_inv"]
        self.sensor = self.model_info["dataset"]["sensor"]
        self.sensor_img_H = self.sensor["img_prop"]["height"]
        self.sensor_img_W = self.sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(self.sensor["img_means"], dtype=torch.float)
        self.sensor_img_stds = torch.tensor(self.sensor["img_stds"], dtype=torch.float)
        self.sensor_fov_up = self.sensor["fov_up"]
        self.sensor_fov_down = self.sensor["fov_down"]
        self.max_points = self.model_info["dataset"]["max_points"]

        self.xentropy_map = []
        self.nclasses = len(self.learning_map_inv)

        self.scan_files = []
        self.label_files = []

        for seq in self.sequences:
            seq = '{0:02d}'.format(int(seq))

            scan_path = os.path.join(self.root, seq, "velodyne")
            label_path = os.path.join(self.root, seq, "labels")

            scan_files = [os.path.join(dp, f) for dp, _, fn in os.walk(os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
            label_files = [os.path.join(dp, f) for dp, _, fn in os.walk(os.path.expanduser(label_path)) for f in fn if is_label(f)]

            if self.gt:
                assert (len(scan_files) == len(label_files))

            self.scan_files.extend(scan_files)
            self.label_files.extend(label_files)

        self.scan_files.sort()
        self.label_files.sort()

        self.set_xentropy_map(self.learning_map)

    def __getitem__(self, index):
        scan_file = self.scan_files[index]
        if self.gt:
            label_file = self.label_files[index]

        DA = False
        flip_sign = False
        rot = False
        drop_points = False
        if self.transform:
            if random.random() > 0.5:
                if random.random() > 0.5:
                    DA = True
                if random.random() > 0.5:
                    flip_sign = True
                if random.random() > 0.5:
                    rot = True
                drop_points = random.uniform(0, 0.5)

        if self.gt:
            scan = SemLaserScan(self.color_map,
                                project=True,
                                H=self.sensor_img_H,
                                W=self.sensor_img_W,
                                fov_up=self.sensor_fov_up,
                                fov_down=self.sensor_fov_down,
                                DA=DA,
                                flip_sign=flip_sign,
                                drop_points=drop_points)
        else:
            scan = LaserScan(project=True,
                             H=self.sensor_img_H,
                             W=self.sensor_img_W,
                             fov_up=self.sensor_fov_up,
                             fov_down=self.sensor_fov_down,
                             DA=DA,
                             rot=rot,
                             flip_sign=flip_sign,
                             drop_points=drop_points)

        scan.open_scan(scan_file)
        if self.gt:
            scan.open_label(label_file)
            scan.sem_label = self.map(scan.sem_label, self.learning_map)
            scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
        if self.gt:
            unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
        else:
            unproj_labels = []

        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            proj_labels = proj_labels * proj_mask
        else:
            proj_labels = []
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                          proj_xyz.clone().permute(2, 0, 1),
                          proj_remission.unsqueeze(0).clone()])
        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        proj = proj * proj_mask.float()

        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

        return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points

    def __len__(self):
        return len(self.scan_files)

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
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
        # do the mapping
        return lut[label]

    def get_sequence(self):
        return self.sequences

    def set_xentropy_map(self, mapdict):
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
            self.xentropy_map = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            self.xentropy_map = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                self.xentropy_map[key] = data
            except IndexError:
                print("Wrong key ", key)

    # to_xentropy 대체하는 함수
    def get_xentropy_map(self, label):
        return self.xentropy_map[label]

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def get_num_classes(self):
        return self.nclasses

def get_loader(dataset, batch_size, shuffle=True, num_worker=0):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_worker,
        pin_memory=True,
        sampler=None
        )
    return dataloader



#
# model_info = yaml.safe_load(open("../UtilityManagement/" + cf.paths["model_info"], 'r'))
# dataset_info = yaml.safe_load(open("../UtilityManagement/" + cf.paths["dataset_info"], 'r'))
#
# dataset = SemanticKitti(model_info, dataset_info, True, 0)
# data_loader = get_loader(dataset, 2, shuffle=True)
#
# for i, (in_vol, proj_mask, proj_labels, _, path_seq, path_name, _, _, _, _, _, _, _, _, _) in enumerate(data_loader):
#     input_point = in_vol[0]
#
#     plt.figure()
#     plt.imshow(input_point[1, :, :])
#     plt.show()
