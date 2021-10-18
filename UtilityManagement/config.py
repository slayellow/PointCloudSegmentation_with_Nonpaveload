import numpy as np


# 데이터셋 관련 경로
paths = dict(
        dataset_info = "parser/semantic-kitti.yaml",
        model_info = "parser/salsanext.yml",
        pretrained_path = "/home/HONG/PretrainedParameter/PointCloudSegmentation",
        save_path = "/home/HONG/Semantic-KITTI/Result",
        log_path = "/home/HONG/Semantic-KITTI/2020-11-18_09-53-12.velodyne32",
        log_save_path = "/home/HONG/Semantic-KITTI/Changwon_Nonpaveload/dataset/sequences/95/velodyne",
        changwon_save_path = "/home/HONG/Semantic-KITTI/Changwon_Nonpaveload/dataset/sequences/95/labels"
)

# paths = dict(
#         dataset_info = "parser/semantic-kitti.yaml",
#         model_info = "parser/salsanext.yml",
#         pretrained_path = "/Users/jinseokhong/data/SemanticKITTI/Result",
#         save_path = "/Users/jinseokhong/data/SemanticKITTI/Result",
#         log_path = "/Users/jinseokhong/data/Changwon_Nonpaveload/2020-11-18_09-53-12.velodyne128xyz",
#         log_save_path = "/Users/jinseokhong/data/Changwon_Nonpaveload/dataset/sequences/99/velodyne",
#         changwon_save_path = "/Users/jinseokhong/data/Changwon_Nonpaveload/Result/"
# )

