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
import pickle

class Brain(Dataset):
    def __init__(self, is_train=True, resize=256, cropsize=224, test_id=1):
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize

        if is_train:
            with open('./content/mnist_shifted_dataset/train_normal.pkl', 'rb') as f:
                normal_train = pickle.load(f)
            self.images = normal_train['images']
            self.images = random.sample(self.images, 1400)
            self.labels = [0] * len(self.images)
        else:
            if test_id == 1:
                with open('./content/mnist_shifted_dataset/test_normal_main.pkl', 'rb') as f:
                    normal_test = pickle.load(f)
                with open('./content/mnist_shifted_dataset/test_abnormal_main.pkl', 'rb') as f:
                    abnormal_test = pickle.load(f)

                normal_test['images'] = random.sample(normal_test['images'], 500)
                abnormal_test['images'] = random.sample(abnormal_test['images'], 500)

                self.images = normal_test['images'] + abnormal_test['images']
                self.labels = [0] * len(normal_test['images']) + [1] * len(abnormal_test['images'])
            else:
                with open('./content/mnist_shifted_dataset/test_normal_shifted.pkl', 'rb') as f:
                    normal_test = pickle.load(f)
                with open('./content/mnist_shifted_dataset/test_abnormal_shifted.pkl', 'rb') as f:
                    abnormal_test = pickle.load(f)
                normal_test['images'] = random.sample(normal_test['images'], 500)
                abnormal_test['images'] = random.sample(abnormal_test['images'], 500)
                self.images = normal_test['images'] + abnormal_test['images']
                self.labels = [0] * len(normal_test['images']) + [1] * len(abnormal_test['images'])



        self.transform = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(cropsize),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        x = self.images[idx]
        # x = Image.open(x).convert('RGB')
        x= Image.fromarray(x.transpose(1, 2, 0))  # Assuming x is in the format (C, H, W)

        x = self.transform(x)

        y = self.labels[idx]

        return x, y, 'None'