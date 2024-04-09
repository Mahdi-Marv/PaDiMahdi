import os
# import tarfile
from PIL import Image
from tqdm import tqdm
# import urllib.request

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from glob import glob
import pandas as pd
import random

# URL = 'ftp://guest:GU.205dldo@ftp.softronics.ch/mvtec_anomaly_detection/mvtec_anomaly_detection.tar.xz'
CLASS_NAMES = [1, 2]


class MVTecDataset(Dataset):
    def __init__(self, dataset_path='D:/dataset/mvtec_anomaly_detection', class_name= 1, is_train=True,
                 resize=256, cropsize=224):
        assert class_name in CLASS_NAMES, 'class_name: {}, should be in {}'.format(class_name, CLASS_NAMES)
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        # self.mvtec_folder_path = os.path.join(root_path, 'mvtec_anomaly_detection')

        # download dataset if not exist
        # self.download()

        # load dataset
        # self.x, self.y, self.mask = self.load_dataset_folder()
        if is_train:
            self.x = glob('/kaggle/input/isic-task3-dataset/dataset/train/NORMAL/*')
            self.y = [0] * len(self.x)
        else:
            test_anomaly_path = glob('/kaggle/input/isic-task3-dataset/dataset/test/ABNORMAL/*')
            test_anomaly_label = [1] * len(test_anomaly_path)
            test_normal_path = glob('/kaggle/input/isic-task3-dataset/dataset/test/NORMAL/*')
            test_normal_label = [0] * len(test_normal_path)

            self.y = test_anomaly_label + test_normal_label
            self.x = test_anomaly_path + test_normal_path

        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        file, y = self.x[idx], self.y[idx]


        x = Image.open(file).convert('RGB')
        x = self.transform_x(x)

        return x, y, ''

    def __len__(self):
        return len(self.x)


def test_loader_2():
    df = pd.read_csv('/kaggle/input/pad-ufes-20/PAD-UFES-20/metadata.csv')

    shifted_test_label = df["diagnostic"].to_numpy()
    shifted_test_label = (shifted_test_label != "NEV")

    shifted_test_path = df["img_id"].to_numpy()
    shifted_test_path = '/kaggle/input/pad-ufes-20/PAD-UFES-20/Dataset/' + shifted_test_path

    shifted_test_set = PAD_UFES_20(image_path=shifted_test_path, labels=shifted_test_label)

    return shifted_test_set


class PAD_UFES_20(Dataset):
    def __init__(self, image_path, labels, count=-1):
        self.image_files = image_path
        self.labels = labels
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

        self.transform = T.Compose([T.Resize(256, Image.ANTIALIAS),
                                    T.CenterCrop(224),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index], ''

    def __len__(self):
        return len(self.image_files)
