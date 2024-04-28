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

class Aptos(Dataset):
    def __init__(self, is_train=True, resize=256, cropsize=224, test_id=1):
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        if is_train:
            self.image_paths = glob('/kaggle/working/APTOS/train/NORMAL/*')

            print('train set size: ', len(self.image_paths))

            # self.image_paths = random.sample(self.image_paths, 1500)

            print('sampled train set size: ', len(self.image_paths))

            self.test_label = [0] * len(self.image_paths)

        else:
            if test_id == 1:
                test_normal_path = glob('/kaggle/working/APTOS/test/NORMAL/*')
                test_anomaly_path = glob('/kaggle/working/APTOS/test/ABNORMAL/*')

                print('len test1 normal: ', len(test_normal_path))
                print('len test1 anomaly: ', len(test_anomaly_path))

                test_normal_path = random.sample(test_normal_path, 600)
                test_anomaly_path = random.sample(test_anomaly_path, 600)



                self.image_paths = test_normal_path + test_anomaly_path

                print('len test1 set: ', len(self.image_paths))

                self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)
            else:
                df = pd.read_csv('/kaggle/input/ddrdataset/DR_grading.csv')
                df = df.sample(n=1000)
                label = df["diagnosis"].to_numpy()
                path = df["id_code"].to_numpy()

                normal_path = path[label == 0]
                anomaly_path = path[label != 0]

                shifted_test_path = list(normal_path) + list(anomaly_path)
                shifted_test_label = [0] * len(normal_path) + [1] * len(anomaly_path)

                shifted_test_path = ["/kaggle/input/ddrdataset/DR_grading/DR_grading/" + s for s in shifted_test_path]

                self.image_paths= shifted_test_path
                print('shift test size: ', len(self.image_paths))
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