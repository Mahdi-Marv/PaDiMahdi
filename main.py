import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from tqdm import tqdm
from collections import OrderedDict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.covariance import LedoitWolf
from scipy.spatial.distance import mahalanobis
from scipy.ndimage import gaussian_filter
from skimage import morphology
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, resnet18
import pandas as pd
from datasets.wbc import WBC_dataset

import datasets.mvtec as mvtec
# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
random.seed(10)
torch.manual_seed(10)
if use_cuda:
    torch.cuda.manual_seed_all(10)

idx = torch.tensor(sample(range(0, 448), 100))


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument("--data_path", type=str, default="D:/dataset/mvtec_anomaly_detection")
    parser.add_argument("--save_path", type=str, default="./wbc_result")
    return parser.parse_args()


def main():

    args = parse_args()

    # load model
    # model = wide_resnet50_2(pretrained=True, progress=True)
    model = resnet18(pretrained=True, progress=True)
    model.to(device)
    model.eval()
    # set model's intermediate outputs
    outputs = []

    def hook(module, input, output):
        outputs.append(output)

    model.layer1[-1].register_forward_hook(hook)
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)

    os.makedirs(os.path.join(args.save_path, 'temp'), exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig_img_rocauc = ax[0]

    df = pd.read_csv('wbc/segmentation_WBC-master/Class Labels of Dataset 1.csv')

    df = df[df['class label'] != 5]
    train_data = df[df['class label'] == 1].sample(n=154, random_state=12)

    df = df.drop(train_data.index)

    test_data = pd.DataFrame()
    for label in [1, 2, 3, 4]:
        class_samples = df[df['class label'] == label].sample(n=22, random_state=1)
        test_data = pd.concat([test_data, class_samples])

    total_roc_auc = []

    class_name = 'one'

    train_data.to_csv('train_dataset.csv', index=False)
    test_data.to_csv('test_dataset.csv', index=False)

    # Use the CustomDataset class
    train_dataset = WBC_dataset(csv_path='train_dataset.csv', image_path='wbc/segmentation_WBC-master/Dataset 1')
    test_dataset = WBC_dataset(csv_path='test_dataset.csv', image_path='wbc/segmentation_WBC-master/Dataset 1', phase='test')

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    # extract train set features
    train_feature_filepath = os.path.join(args.save_path, 'temp', 'train_%s.pkl' % class_name)
    if not os.path.exists(train_feature_filepath):
        print('feature path does not exist')
        for (x, _) in tqdm(train_dataloader, '| feature extraction | train | %s |' % class_name):
            # model prediction
            with torch.no_grad():
                _ = model(x.to(device))
            # get intermediate layer outputs
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v.cpu().detach())
            # initialize hook outputs
            outputs = []
        for k, v in train_outputs.items():
            train_outputs[k] = torch.cat(v, 0)

        # Embedding concat
        embedding_vectors = train_outputs['layer1']
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name])

        # randomly select 100 dimension
        embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.size()
        embedding_vectors = embedding_vectors.view(B, C, H * W)
        mean = torch.mean(embedding_vectors, dim=0).numpy()
        cov = torch.zeros(C, C, H * W).numpy()
        for i in range(H * W):
            cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_ + 0.001
        # save learned distribution
        train_outputs = [mean, cov]
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_outputs, f)
    else:
        print('load train set feature from: %s' % train_feature_filepath)
        with open(train_feature_filepath, 'rb') as f:
            train_outputs = pickle.load(f)

    gt_list = []
    # gt_mask_list = []
    test_imgs = []

    # extract test set features
    for (x, y) in tqdm(test_dataloader, '| feature extraction | test | %s |' % class_name):
        test_imgs.extend(x.cpu().detach().numpy())
        gt_list.extend(y.cpu().detach().numpy())
        # gt_mask_list.extend(mask.cpu().detach().numpy())
        # model prediction
        with torch.no_grad():
            _ = model(x.to(device))
        # get intermediate layer outputs
        for k, v in zip(test_outputs.keys(), outputs):
            test_outputs[k].append(v.cpu().detach())
        # initialize hook outputs
        outputs = []
    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)
    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

    # randomly select 100 dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
    # calculate distance matrix
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W).detach().numpy()
    dist_list = []
    for i in range(H * W):
        mean = train_outputs[0][:, i]
        conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
        dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
        dist_list.append(dist)

    dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

    # upsample
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=x.size(2), mode='bilinear',
                              align_corners=False).squeeze().numpy()
    # apply gaussian smoothing on the score map
    for i in range(score_map.shape[0]):
        score_map[i] = gaussian_filter(score_map[i], sigma=4)
    # Normalization
    max_score = score_map.max()
    min_score = score_map.min()
    scores = (score_map - min_score) / (max_score - min_score)
    # calculate image-level ROC AUC score
    img_scores = scores.reshape(scores.shape[0], -1).max(axis=1)
    gt_list = np.asarray(gt_list)
    fpr, tpr, _ = roc_curve(gt_list, img_scores)
    img_roc_auc = roc_auc_score(gt_list, img_scores)
    total_roc_auc.append(img_roc_auc)
    print('image ROCAUC: %.3f' % (img_roc_auc))


    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")






def denormalization(x):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = (((x.transpose(1, 2, 0) * std) + mean) * 255.).astype(np.uint8)
    return x


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


if __name__ == '__main__':
    main()
