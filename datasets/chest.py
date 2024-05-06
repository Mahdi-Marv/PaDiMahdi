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
import pydicom

class Chest(Dataset):
    def __init__(self, is_train=True, resize=256, cropsize=224, test_id=1):
        self.is_train = is_train
        self.resize = resize
        self.cropsize = cropsize
        self.test_id = test_id

        if is_train:
            self.image_paths = glob('/kaggle/working/train/normal/*')
            self.image_paths = random.sample(self.image_paths, 2000)
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
                test_normal_path = glob('/kaggle/working/test/normal/*')
                test_anomaly_path = glob('/kaggle/working/test/anomaly/*')

                print('len test1 normal: ', len(test_normal_path))
                print('len test1 anomaly: ', len(test_anomaly_path))

                test_normal_path = random.sample(test_normal_path, 500)
                test_anomaly_path = random.sample(test_anomaly_path, 500)



                self.image_paths = test_normal_path + test_anomaly_path

                print('len test1 set: ', len(self.image_paths))

                self.test_label = [0] * len(test_normal_path) + [1] * len(test_anomaly_path)
            else:
                test_normal_path = glob('/kaggle/working/4. Operations Department/Test/1/*')
                test_anomaly_path = (glob('/kaggle/working/4. Operations Department/Test/0/*') + glob(
                    '/kaggle/working/4. Operations Department/Test/2/*') +
                                             glob('/kaggle/working/4. Operations Department/Test/3/*'))

                print('len test1 normal: ', len(test_normal_path))
                print('len test1 anomaly: ', len(test_anomaly_path))

                # test_normal_path = random.sample(test_normal_path, 450)
                # test_anomaly_path = random.sample(test_anomaly_path, 450)

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
        if self.test_id==1 or self.is_train:
            dicom = pydicom.dcmread(self.image_paths[idx])
            image = dicom.pixel_array
            image = Image.fromarray(image).convert('RGB')
            image = self.transform(image)
            y = self.test_label[idx]
            return image, y, 'none'
        x = self.image_paths[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform(x)

        y = self.test_label[idx]

        return x, y, 'None'