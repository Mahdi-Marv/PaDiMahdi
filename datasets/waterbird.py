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

class Waterbird(Dataset):
    def __init__(self, is_train=True, resize=256, cropsize=224, mode='sd', count_train_landbg=1, count_train_waterbg=1):
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        copy = False
        root = '/kaggle/input/waterbird/waterbird'
        df = pd.read_csv(os.path.join(root, 'metadata.csv'))

        print(len(df))

        train = is_train
        self.df = df
        lb_on_l = df[(df['y'] == 0) & (df['place'] == 0)]
        lb_on_w = df[(df['y'] == 0) & (df['place'] == 1)]
        self.normal_paths = []
        self.labels = []

        normal_df = lb_on_l.iloc[:count_train_landbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_landbg])
        normal_df = lb_on_w.iloc[:count_train_waterbg]
        normal_df_np = normal_df['img_filename'].to_numpy()
        copy_count = 1
        if copy:
            copy_count = count_train_landbg // count_train_waterbg
        for _ in range(copy_count):
            self.normal_paths.extend([os.path.join(root, x) for x in normal_df_np][:count_train_waterbg])

        if train:
            self.image_paths = self.normal_paths
            self.labels = [0] * len(self.image_paths)
        else:
            self.image_paths = []
            if mode == 'bg_all':
                dff = df
            elif mode == 'bg_water':
                dff = df[(df['place'] == 1)]
            elif mode == 'bg_land':
                dff = df[(df['place'] == 0)]
            elif mode == 'ood':
                dff = df[(df['place'] == 0) & (df['y'] == 1)]
            else:
                print('Wrong mode!')
                raise ValueError('Wrong bg mode!')
            all_paths = dff[['img_filename', 'y']].to_numpy()
            for i in range(len(all_paths)):
                full_path = os.path.join(root, all_paths[i][0])
                if full_path not in self.normal_paths:
                    self.image_paths.append(full_path)
                    self.labels.append(all_paths[i][1])

        if not is_train:
            indices = list(range(len(self.image_paths)))
            random.shuffle(indices)
            selected_indices = indices[:1400]
            selected_paths = []
            selected_labels = []
            for i in selected_indices:
                selected_paths.append(self.image_paths[i])
                selected_labels.append(self.labels[i])

            self.image_paths = selected_paths
            self.labels = selected_labels

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

        y = self.labels[idx]

        return x, y, 'None'
def get_waterbird_trainset():
    train_set = Waterbird(is_train=True, count_train_landbg=1300, count_train_waterbg=100, mode='bg_all')

    return train_set


def get_waterbird_test_set():
    test_set = Waterbird(is_train=False, count_train_landbg=1300, count_train_waterbg=100, mode='bg_land')
    return test_set

def get_waterbird_just_test_shifted():
    test_set = Waterbird(is_train=False, count_train_landbg=1300, count_train_waterbg=100, mode='bg_water')
    return test_set
