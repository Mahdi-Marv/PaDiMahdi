import os
# import tarfile
from PIL import Image
from tqdm import tqdm
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from glob import glob

class Brain(Dataset):
    def __init__(self, is_train=True, resize=256, cropsize=224, test_id=1):
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        if is_train:
            self.image_paths = glob('./Br35H/dataset/train/normal/*')
            self.test_label = [0] * len(self.image_paths)

        else:
            if test_id == 1:
                test_normal_path = glob('./Br35H/dataset/test/normal/*')
                test_anomaly_path = glob('./Br35H/dataset/test/anomaly/*')

                self.image_paths = test_normal_path + test_anomaly_path
                self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)
            else:
                test_normal_path = glob('./brats/dataset/test/normal/*')
                test_anomaly_path = glob('./brats/dataset/test/anomaly/*')

                self.image_paths = test_normal_path + test_anomaly_path
                self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)

        self.transform = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        x = self.image_paths[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        y = self.test_label[idx]

        return x, y, None