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
import pandas as pd

class Isic(Dataset):
    def __init__(self, is_train=True, resize=256, cropsize=224, test_id=1):
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        if is_train:
            self.image_paths = glob('/kaggle/input/isic-task3-dataset/dataset/train/NORMAL/*')

            print('train set size: ', len(self.image_paths))

            self.image_paths = random.sample(self.image_paths, 2000)

            print('sampled train set size: ', len(self.image_paths))

            self.test_label = [0] * len(self.image_paths)

        else:
            if test_id == 1:
                test_normal_path = glob('/kaggle/input/isic-task3-dataset/dataset/test/NORMAL/*')
                test_anomaly_path = glob('/kaggle/input/isic-task3-dataset/dataset/test/ABNORMAL/*')

                print('len test1 normal: ', len(test_normal_path))
                print('len test1 anomaly: ', len(test_anomaly_path))

                # test_normal_path = random.sample(test_normal_path, 450)
                # test_anomaly_path = random.sample(test_anomaly_path, 450)



                self.image_paths = test_normal_path + test_anomaly_path

                print('len test1 set: ', len(self.image_paths))

                self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)
            else:
                df = pd.read_csv('/kaggle/input/pad-ufes-20/PAD-UFES-20/metadata.csv')

                shifted_test_label = df["diagnostic"].to_numpy()
                shifted_test_label = (shifted_test_label != "NEV")

                shifted_test_path = df["img_id"].to_numpy()
                shifted_test_path = '/kaggle/input/pad-ufes-20/PAD-UFES-20/Dataset/' + shifted_test_path

                self.image_paths = shifted_test_path
                print('len test2 set: ', len(self.image_paths))
                self.test_label = shifted_test_label

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