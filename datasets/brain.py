import os
# import tarfile
from PIL import Image
from tqdm import tqdm
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from glob import glob
import random

class Brain(Dataset):
    def __init__(self, is_train=True, resize=256, cropsize=224, test_id=1):
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        if is_train:
            node0_train = glob('/kaggle/input/camelyon17-clean/node0/train/normal/*')
            node1_train = glob('/kaggle/input/camelyon17-clean/node1/train/normal/*')
            node2_train = glob('/kaggle/input/camelyon17-clean/node2/train/normal/*')

            self.image_paths = node0_train + node1_train + node2_train
            self.image_paths = random.sample(self.image_paths, 3000)
            # brats_mod = glob('./brats/dataset/train/normal/*')
            #
            # random.seed(1)
            #
            # random_brats_images = random.sample(brats_mod, 150)
            #
            # print('added 150 normal brat images')
            #
            # self.image_paths.extend(random_brats_images)
            self.test_label = [0] * len(self.image_paths)

        else:
            if test_id == 1:
                node0_test_normal = glob('/kaggle/input/camelyon17-clean/node0/test/normal/*')
                node0_test_anomaly = glob('/kaggle/input/camelyon17-clean/node0/test/anomaly/*')

                node1_test_normal = glob('/kaggle/input/camelyon17-clean/node1/test/normal/*')
                node1_test_anomaly = glob('/kaggle/input/camelyon17-clean/node1/test/anomaly/*')

                node2_test_normal = glob('/kaggle/input/camelyon17-clean/node2/test/normal/*')
                node2_test_anomaly = glob('/kaggle/input/camelyon17-clean/node2/test/anomaly/*')

                test_normal_path = node0_test_normal + node1_test_normal + node2_test_normal
                test_anomaly_path = node0_test_anomaly + node1_test_anomaly + node2_test_anomaly

                print('len test1 normal: ', len(test_normal_path))
                print('len test1 anomaly: ', len(test_anomaly_path))

                test_normal_path = random.sample(test_normal_path, 500)
                test_anomaly_path = random.sample(test_anomaly_path, 500)



                self.image_paths = test_normal_path + test_anomaly_path

                print('len test1 set: ', len(self.image_paths))

                self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)
            else:
                node3_test_normal = glob('/kaggle/input/camelyon17-clean/node3/test/normal/*')
                node3_test_anomaly = glob('/kaggle/input/camelyon17-clean/node3/test/anomaly/*')

                node4_test_normal = glob('/kaggle/input/camelyon17-clean/node4/test/normal/*')
                node4_test_anomaly = glob('/kaggle/input/camelyon17-clean/node4/test/anomaly/*')

                test_normal_path = node3_test_normal + node4_test_normal
                test_anomaly_path = node3_test_anomaly + node4_test_anomaly

                print('len test1 normal: ', len(test_normal_path))
                print('len test1 anomaly: ', len(test_anomaly_path))

                test_normal_path = random.sample(test_normal_path, 500)
                test_anomaly_path = random.sample(test_anomaly_path, 500)

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
        x = self.transform(x)

        y = self.test_label[idx]

        return x, y, 'None'