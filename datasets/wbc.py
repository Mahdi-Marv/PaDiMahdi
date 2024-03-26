import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision import transforms as T

class WBC_dataset(Dataset):
    def __init__(self, image_path="", csv_path="", resize=224, class_name=1, phase='train'):
        self.path = image_path
        self.resize = resize
        self.class_name = class_name
        self.phase = phase
        self.img_labels = pd.read_csv(csv_path)
        self.transform = T.Compose([T.Resize(resize, Image.ANTIALIAS),
                                      T.CenterCrop(224),
                                      T.ToTensor(),
                                      T.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        img_path = f"{self.path}/{str(self.img_labels.iloc[idx, 0]).zfill(3)}.png"
        # print(img_path)
        image = Image.open(img_path).convert('RGB')
        label = self.img_labels.iloc[idx, 1]

        image = self.transform(image)

        clas = 0 if label == 1 else 1
        # if self.phase == 'test':
        #     print(clas)
        return image, clas

    def __len__(self):
        return len(self.img_labels)

    def transform(self, img):
        pass
