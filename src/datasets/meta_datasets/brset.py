import os
import copy
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms, datasets

from src.datasets.root_paths import DATA_ROOTS


import os
import torch
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations
import cv2

from PIL import Image






class RETINAL(Dataset):
    def __init__(self, train = True,root = 'D:/Course/598-007/project/viewmaker-privacy-yhr/a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.0/', image_transforms = None):
        class_column1 = 'Class'
        class_column2 = 'patient_sex'

        if train:
            df_subset = pd.read_csv(root+"/train.csv")
        else:
            df_subset = pd.read_csv(root+'/val.csv')
        self.studyuid = df_subset["image_id"].astype(str).values
        if isinstance(df_subset[class_column1][0], str):
            self.labels_1 = df_subset[class_column1].astype('category').cat.codes.values
        else:
            self.labels_1 = df_subset[class_column1].values
        if isinstance(df_subset[class_column2][0], str):
            self.labels_2 = df_subset[class_column2].astype('category').cat.codes.values
        else:
            self.labels_2 = df_subset[class_column2].values
        self.transform = image_transforms
        self.retinal_path = root+'fundus_photos/'

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
        return idx, image, labels1, labels2-1

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





if __name__ == '__main__':
    data=RETINAL()
    pass