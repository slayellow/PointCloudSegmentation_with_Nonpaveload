import numpy as np


# 데이터셋 관련 경로

# paths = dict(
#     img_dir = "/home/HONG/GTA5_Dataset/img_dir",
#     ann_dir = "/home/HONG/GTA5_Dataset/ann_dir",
#     train_dataset_file = "/home/HONG/ImageSegmentation_with_NonPaveload/DataManagement/train.txt",
#     valid_dataset_file = "/home/HONG/ImageSegmentation_with_NonPaveload/DataManagement/valid.txt",
#     pretrained_path = "/home/HONG/PretrainedParameter",
# )

paths = dict(
        img_dir = "/Users/jinseokhong/data/GTA5Dataset/img_dir",
        ann_dir = "/Users/jinseokhong/data/GTA5Dataset/ann_dir",
        train_dataset_file = "/Users/jinseokhong/PycharmProjects/ImageSegmentation_with_NonPaveload/DataManagement/train.txt",
        valid_dataset_file = "/Users/jinseokhong/PycharmProjects/ImageSegmentation_with_NonPaveload/DataManagement/valid.txt",
        pretrained_path = "/Users/jinseokhong/data/GTA5Dataset",
)

IMG_WIDTH = 480
IMG_HEIGHT = 640
NUM_CLASSES = 19
palette = np.array([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32]
    ])

# 네트워크 구성 관련 파라메타

network_info = dict(
    batch_size = 2,                        # batch_size take during training
    epochs = 100,                            # total number of epoch
    learning_rate = 0.1,                   # learining rate
    freq_print = 10,
    num_worker = 0
)
