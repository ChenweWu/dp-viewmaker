# import relevant packages
from functools import lru_cache
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, SGD
# from torchvision import transforms
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from albumentations import Resize
from albumentations.pytorch import ToTensorV2
import albumentations
import cv2
import random
import json
import zipfile
from torchvision import transforms
# import pydicom as dicom
from PIL import Image


class RETINAL(Dataset):
    def __init__(self, df_all, retinal_path, class_column1, class_column2, transform):
        # df_subset = df_all[df_all["image_id"].isin(df_studyIDs[0])]
        df_subset = df_all
        self.studyuid = df_subset["image_id"].astype(str).values
        if isinstance(df_subset[class_column1][0], str):
            self.labels_1 = df_subset[class_column1].astype('category').cat.codes.values
        else:
            self.labels_1 = df_subset[class_column1].values
        if isinstance(df_subset[class_column2][0], str):
            self.labels_2 = df_subset[class_column2].astype('category').cat.codes.values
        else:
            self.labels_2 = df_subset[class_column2].values
        self.transform = transform
        self.retinal_path = retinal_path

    def __len__(self):
        return self.studyuid.shape[0]

    def __getitem__(self, idx):
        path = self.studyuid[idx]
        # print(path)
        path = os.path.join(self.retinal_path, "img['img_" + path + ".jpg.pkl']" + ".png")
        # print(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist")
        image = cv2.imread(path)
        image = np.array(image)
        image = self.transform(image=image)['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensorV2()(image=image)["image"]
        labels1 = self.labels_1[idx]
        labels2 = self.labels_2[idx]
        return image, labels1,  labels2-1
    # labels1 disease
    # labels2 sex



def loadRetinalData(train_path, val_path, df_train, df_val, batch_size, image_size):
    train_dataset = RETINAL(df_train, train_path, 'Class', 'patient_sex', transform=get_transform(image_size, 'train'))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataset = RETINAL(df_val, val_path, 'Class', 'patient_sex', transform=get_transform(image_size, 'val'))
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader


def loadTestRetinalData(test_path,  df_test, batch_size, image_size):
    test_dataset = RETINAL(df_test, test_path, 'Class', 'patient_sex', transform=get_transform(image_size, 'train'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


def get_transform(image_size, split):
    transforms_train = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(mean=[0.59088606, 0.2979608, 0.10854383], std=[0.28389975, 0.15797651, 0.06909362],
                                 always_apply=True, max_pixel_value=255.0),
    ])
    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(mean=[0.58375394, 0.29545224, 0.10972021], std=[0.28172594, 0.15677236, 0.069528475],
                                 always_apply=True, max_pixel_value=255.0),
    ])
    if split == 'train':
        return transforms_train
    elif split == 'val':
        return transforms_val
    else:
        raise NotImplementedError


def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    # print('Compute mean and variance for val data.')
    # print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    return list(mean.numpy()), list(std.numpy())


if __name__ == '__main__':

    df_train = pd.read_csv('./train.csv')
    df_val = pd.read_csv('./val.csv')
    df_test = pd.read_csv('./test.csv')
    # 设置图像大小和批量大小
    image_size = 224
    batch_size = 32
    train_loader, valid_loader = loadRetinalData('./data_folder/train', './data_folder/val', df_train, df_val,
                                                 batch_size, image_size)
    test_loader = loadTestRetinalData('./data_folder/test', df_test, batch_size, image_size)
    # print(train_loader.__len__())
    # print(valid_loader.__len__())
    # print("Training Data:")
    # for batch_idx, (images, labels1, labels2) in enumerate(train_loader):
    #     print(f"Batch {batch_idx + 1}:")
    #     print("Images shape:", images.shape)
    #     print("Labels1 shape:", labels1.shape)
    #     print("Labels2 shape:", labels2.shape)
    #     print(batch_idx)
    #     image = Image.fromarray(images[0].permute(1, 2, 0).numpy().astype(np.uint8))
    #     image.show()
    #
    # print("Validation Data:")
    # for batch_idx, (images, labels1, labels2) in enumerate(valid_loader):
    #     print(f"Batch {batch_idx + 1}:")
    #     print("Images shape:", images.shape)
    #     print("Labels1 shape:", labels1.shape)
    #     print("Labels2 shape:", labels2.shape)
    #     print(batch_idx)
    #     image = Image.fromarray(images[0].permute(1, 2, 0).numpy().astype(np.uint8))
    #     image.show()
