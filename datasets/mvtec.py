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
            self.x = glob('/kaggle/working/Mean-Shifted-Anomaly-Detection/APTOS/train/NORMAL/*')
            self.y = [0] * len(self.x)
        else:
            test_normal_path = glob('/kaggle/working/Mean-Shifted-Anomaly-Detection/APTOS/test/NORMAL/*')
            test_anomaly_path = glob('/kaggle/working/Mean-Shifted-Anomaly-Detection/APTOS/test/ABNORMAL/*')

            self.x = test_normal_path + test_anomaly_path
            self.y = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)


        # set transforms
        self.transform_x = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]

        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        return x, y, ''

    def __len__(self):
        return len(self.x)


def test_loader_2():
    df = pd.read_csv('/kaggle/input/ddrdataset/DR_grading.csv')
    label = df["diagnosis"].to_numpy()
    path = df["id_code"].to_numpy()

    normal_path = path[label == 0]
    anomaly_path = path[label != 0]

    shifted_test_path = list(normal_path) + list(anomaly_path)
    shifted_test_label = [0] * len(normal_path) + [1] * len(anomaly_path)

    shifted_test_path = ["/kaggle/input/ddrdataset/DR_grading/DR_grading/" + s for s in shifted_test_path]
    shifted_test_set = APTOS(image_path=shifted_test_path, labels=shifted_test_label)

    return shifted_test_set


class APTOS(Dataset):
    def __init__(self, image_path, labels, count=-1):
        self.image_files = image_path
        self.labels = labels
        self.transform = T.Compose([T.Resize(256, Image.ANTIALIAS),
                                      T.CenterCrop(224),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])
        if count != -1:
            if count < len(self.image_files):
                self.image_files = self.image_files[:count]
                self.labels = self.labels[:count]
            else:
                t = len(self.image_files)
                for i in range(count - t):
                    self.image_files.append(random.choice(self.image_files[:t]))
                    self.labels.append(random.choice(self.labels[:t]))

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image = Image.open(image_file)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.image_files)
