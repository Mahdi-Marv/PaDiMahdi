from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import torch.utils.data as data
import matplotlib.image as mpimg
from torchvision import transforms
import random
from torchvision import transforms as T
import torch
from PIL import Image
import matplotlib.pyplot as plt

from imagenet_30_dataset import IMAGENET30_TEST_DATASET


def center_paste(large_img, small_img):
    # Calculate the center position
    large_width, large_height = large_img.size
    small_width, small_height = small_img.size

    # Calculate the top-left position
    left = (large_width - small_width) // 2
    top = (large_height - small_height) // 2

    # Create a copy of the large image to keep the original unchanged
    result_img = large_img.copy()

    # Paste the small image onto the large one at the calculated position
    result_img.paste(small_img, (left, top))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # figsize can be adjusted as needed

    # Display each image in its respective subplot
    axs[0].imshow(large_img)
    axs[1].imshow(small_img)
    axs[2].imshow(result_img)
    plt.show()

    return result_img


class MVTEC(data.Dataset):
    """`MVTEC <https://www.mvtec.com/company/research/datasets/mvtec-ad/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directories
            ``bottle``, ``cable``, etc., exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        resize (int, optional): Desired output image size.
        interpolation (int, optional): Interpolation method for downsizing image.
        category: bottle, cable, capsule, etc.
    """

    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 category='carpet', resize=224, interpolation=2, use_imagenet=True,
                 select_random_image_from_imagenet=True, shrink_factor=0.9):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = resize
        self.category = category
        if use_imagenet:
            self.resize = int(resize * shrink_factor)
        self.interpolation = interpolation
        self.select_random_image_from_imagenet = select_random_image_from_imagenet

        self.transform = T.Compose([T.Resize(224, Image.ANTIALIAS),

                                    T.ToTensor(),
                                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])])

        self.transform_mask = T.Compose([T.Resize(resize, Image.NEAREST),
                                         T.CenterCrop(224),
                                         T.ToTensor()])

        # load images for training
        if self.train:
            pass
        else:
            # load images for testing
            self.test_data, self.test_labels, self.mask = self.load_dataset_folder()

    def __getitem__(self, index):
        """
            Args:
                index (int): Index
            Returns:
                tuple: (image, target) where target is index of the target class.
            """

        imagenet30_testset = IMAGENET30_TEST_DATASET()

        img, target, mask = self.test_data[index], self.test_labels[index], self.mask[index]

        img = Image.open(img).convert('RGB')

        if target == 0:
            mask = torch.zeros([1, 224, 224])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)

        if self.select_random_image_from_imagenet:
            imagenet30_img = imagenet30_testset[int(random.random() * len(imagenet30_testset))][0].resize(
                (224, 224))
        else:
            imagenet30_img = imagenet30_testset[100][0].resize((224, 224))

        # if resizing image
        if self.resize is not None:
            resizeTransf = transforms.Resize(self.resize, Image.ANTIALIAS)
            img = resizeTransf(img)

        #         print(f"imagenet30_img.size: {imagenet30_img.size}")
        #         print(f"img.size: {img.size}")
        img = center_paste(imagenet30_img, img)

        # if self.transform is not None:
        #     img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, mask

    def __len__(self):
        """
        Args:
            None
        Returns:
            int: length of array.
        """
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def load_dataset_folder(self):
        x, y, mask = [], [], []

        img_dir = os.path.join(self.root, self.category, 'test')
        gt_dir = os.path.join(self.root, self.category, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:

            # load images
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.png')])
            x.extend(img_fpath_list)

            # load gt labels
            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '_mask.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask)
