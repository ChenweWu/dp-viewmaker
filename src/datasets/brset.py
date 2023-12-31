import os
import copy
from PIL import Image
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms, datasets

from src.datasets.root_paths import DATA_ROOTS
import os
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations
import cv2

from PIL import Image


class RETINAL(Dataset):
    def __init__(self, train=True, root='/home/ubuntu/data/br_images/BR_Snow_orig/',
                 image_size=224, image_transforms=True):
        class_column1 = 'Class'
        class_column2 = 'patient_sex'

        if train:
            df_subset = pd.read_csv(root+"train.csv")
        else:
            df_subset = pd.read_csv(root+'val.csv')

        self.studyuid = df_subset["image_id"].astype(str).values

        if isinstance(df_subset[class_column1][0], str):
            self.labels_1 = df_subset[class_column1].astype('category').cat.codes.values
        else:
            self.labels_1 = df_subset[class_column1].values
        if isinstance(df_subset[class_column2][0], str):
            self.labels_2 = df_subset[class_column2].astype('category').cat.codes.values
        else:
            self.labels_2 = df_subset[class_column2].values
        self.retinal_path = root 
        self.img_size = image_size
        if image_transforms is True:
            self.transform = self.get_transform()

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
        if self.transform is not None:
            image = self.transform(image=image)['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensorV2()(image=image)["image"]
        labels1 = self.labels_1[idx]
        labels2 = self.labels_2[idx]
        return idx, image, labels1, labels2-1

    def get_transform(self):
        transforms = albumentations.Compose([
            albumentations.Resize(self.img_size, self.img_size),
            albumentations.Normalize(mean=[0.59088606, 0.2979608, 0.10854383], std=[0.28389975, 0.15797651, 0.06909362],
                                     always_apply=True, max_pixel_value=255.0),
        ])
        return transforms


if __name__ == '__main__':
    data=RETINAL(image_transforms=True)
    train_loader = DataLoader(data, batch_size=16, shuffle=True)
    print(train_loader.__len__())
    print("Training Data:")
    for batch_idx, (images, labels1, labels2) in enumerate(train_loader):
        print(f"Batch {batch_idx + 1}:")
        print("Images shape:", images.shape)
        print("Labels1 shape:", labels1.shape)
        print("Labels2 shape:", labels2.shape)
        print(batch_idx)
        image = Image.fromarray(images[0].permute(1, 2, 0).numpy().astype(np.uint8))
        image.show()
    pass