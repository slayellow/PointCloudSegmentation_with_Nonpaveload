import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from PIL import Image
import UtilityManagement.config as cf
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpathces


def save_datalist():
    if os.path.isfile(cf.paths['train_dataset_file']):
        print('Training Dataset File is existed')
        if os.path.isfile(cf.paths['valid_dataset_file']):
            print("Validation Dataset File is existed")
            return
    else:
        img_filename_list = os.listdir(cf.paths["img_dir"])
        ann_filename_list = os.listdir(cf.paths["ann_dir"])

        with open(cf.paths['train_dataset_file'], 'w') as f:
            train_length = int(len(img_filename_list) * 4 / 5)
            f.writelines(os.path.join(cf.paths['img_dir'], img) + ' ' + os.path.join(cf.paths['ann_dir'], ann) + '\n'
                         for img, ann in zip(img_filename_list[:train_length], ann_filename_list[:train_length]))

        with open(cf.paths['valid_dataset_file'], 'w') as f:
            f.writelines(os.path.join(cf.paths['img_dir'], img) + ' ' + os.path.join(cf.paths['ann_dir'], ann) + '\n'
                         for img, ann in zip(img_filename_list[train_length:], ann_filename_list[train_length:]))


def decode_segmap(mask):
    label_color = cf.palette

    r = mask.copy()
    g = mask.copy()
    b = mask.copy()
    for label in range(0, cf.NUM_CLASSES):
        r[mask == label] = label_color[label, 0]
        g[mask == label] = label_color[label, 1]
        b[mask == label] = label_color[label, 2]
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0

    return rgb


class GTA5Dataset(Dataset):
    def __init__(self, path):
        self.dataset = np.loadtxt(path, dtype = str)

        self.img = self.dataset[:, 0]
        self.ann = self.dataset[:, 1]

        self.transform = transforms.Compose([
            transforms.Resize((cf.IMG_WIDTH, cf.IMG_HEIGHT), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()])

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1, 34]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
                            'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
                            'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
                            'motorcycle', 'bicycle']

        self.ignore_index = 255

        self.class_map = dict(zip(self.valid_classes, range(cf.NUM_CLASSES)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = Image.open(self.img[idx]).convert('RGB')
        ann = np.array(Image.open(self.ann[idx]), dtype=np.uint8)
        ann = self.encode_segmap(ann)
        target = Image.fromarray(ann)

        data = {'source': self.transform(img), 'target': self.transform(target)}
        return data

    def encode_segmap(self, mask):
        for void_class in self.void_classes:
            mask[mask == void_class] = self.ignore_index
        for valid_class in self.valid_classes:
            mask[mask == valid_class] = self.class_map[valid_class]
        return mask



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
# dataset = GTA5Dataset(cf.paths['train_dataset_file'])
# data_loader = get_loader(dataset, 2, shuffle=True)
#
# for i, batches in enumerate(data_loader):
#     for j in range(batches['source'].size()[0]):
#         img = batches['source'].numpy()
#         ann = batches['target'].numpy()
#         ann *= 255.0
#         target = np.array(ann[j]).astype(np.uint8)
#         segmap = decode_segmap(target)
#         source = np.transpose(img[j], axes = [1, 2, 0])
#
#         plt.figure()
#         plt.subplot(211)
#         plt.imshow(source)
#         plt.subplot(212)
#         plt.imshow(segmap)
#         plt.show()
