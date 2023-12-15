#!/usr/bin/env python

# import relevant packages

# define functions and classes only relevant to training
import os
import random
from typing import Mapping, Any

import dotmap
import numpy as np
from dotmap import DotMap
from collections import OrderedDict

from sklearn.metrics import accuracy_score, f1_score,confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

from src.datasets import datasets
from src.models import resnet_small, resnet, models
from src.models.transfer import LogisticRegression
from src.objectives.memory_bank import MemoryBank
from src.objectives.adversarial import  AdversarialSimCLRLoss,  AdversarialNCELoss
from src.objectives.infonce import NoiseConstrastiveEstimation
from src.objectives.simclr import SimCLRObjective
from src.objectives.focal import FocalLoss
from src.utils import utils
from src.datasets.brset import RETINAL
from src.models import viewmaker

import torch_dct as dct
import pytorch_lightning as pl
# import wandb
import timm
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from torchvision.utils import save_image
class ResNet200D(nn.Module):
    def __init__(self, n_classes, model_name='resnet200d'):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        n_features = self.model.fc.in_features
        self.model.global_pool = nn.Identity()
        self.model.fc = nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        bs = x.size(0)
        features = self.model(x)
        pooled_features = self.pooling(features).view(bs, -1)
        output = self.fc(pooled_features)
        return output
def main(config):
    """Trains a model using a config file

    Parameters
    ----------
    config_path : str, optional
        The path to the config file (default is train.conf)

    Returns
    -------
    thing_to_return
        description of thing to return
    """
    # DEVICE
    DEVICE = config['device']

     # Seed
    seed_everything(config['seed'])
    
    # Load data
    df_all = readData(config['train_path'])
    df_val = readTestData(config['test_path'])
    # val_perc = None if config['val_perc']=="None" else config['val_perc']
    # folds = k_fold_cross_val(df_train_val, df_all, k=config['k'], stratified_grouped = config['stratified_grouped'], val_perc=val_perc)
    
    # for curr_fold in range(len(folds)):
    #     print('Training on Fold ' + str(curr_fold + 1) + ' of ' + str(len(folds)))
    train_loader = loadRetinalData2( df_all, config['batch_size'], config['image_size'], config['retinal_path'], config['class_column'], config['channel_avg'], config['channel_std'], config['crop_dims'], split='train', num_workers=config['num_loader_workers'])
    val_loader = loadRetinalData2( df_val, 1, config['image_size'], config['retinal_path'], config['class_column'], config['channel_avg'], config['channel_std'], config['crop_dims'], split='val', num_workers=config['num_loader_workers'])
    # Model

    model = ResNet200D(3)
    view_model = viewmaker.Viewmaker(
        num_channels=3,
        distortion_budget=0.7,
        activation='relu',
        clamp=False,
        frequency_domain=False,
        downsample_to=False,
        num_res_blocks=3,
    )
    # view_model=view_model.load_state_dict(torch.load('/home/ubuntu/vmkepoch5f1 0.6156336573680006'))
    view_model.load_state_dict(torch.load('/home/ubuntu/vmkstd4f1 0.6601222145115274'))
    view_model.to(DEVICE)
    model = model.to(DEVICE)
    # Loss Fc
    class_dis = np.array(config['class_dist'])
    class_weights =1-class_dis/np.sum(class_dis)
    print(class_weights)
    criterion = get_lossfn(config['loss'],torch.tensor(class_weights).float().to(DEVICE))

    # Get optim
    optimizer = get_optim(model, config['optimizer'], config['lr'])

    # Train
    best_loss = np.inf
    best_f1 = 0
    valid_loss = 0
    NUM_EPOCH = config['NUM_EPOCH']
    for epoch in range(NUM_EPOCH):
        model.eval()
        print(f"Start training EPOCH {epoch}...")
        y_list = []
        pred_list = []
        i = 0
        for X, y,fn in tqdm(val_loader,total=len(val_loader)):
            optimizer.zero_grad()
            X = X.float().to(DEVICE)
            y = y.long().to(DEVICE)

            
            X2 = view_model(X)
            # print(X.shape)
            i+=1
            save_image(X[0], f'/home/ubuntu/eg/{i}_orig.png')
            save_image(X2[0], f'/home/ubuntu/eg/{i}.png')
            
        break

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
# import pydicom as dicom

# RETINAL_PATH = "/home/ubuntu/dp_snow_0_3/"

class RETINAL(Dataset):
    """
    Brazilian Retinal Image Dataset Class
    """
    def __init__(self, df_all, retinal_path, class_column, transform=None):
     #   df_subset = df_all[df_all["image_id"].isin(df_studyIDs[0])]
        df_subset = df_all
        self.studyuid = df_subset["image_id"].astype(str).values
        if isinstance(df_subset[class_column][0], str):
            self.labels = df_subset[class_column].astype('category').cat.codes.values
        else:
            self.labels = df_subset[class_column].values
        self.transform = transform
        self.retinal_path = retinal_path
        
    def __len__(self):
        return self.studyuid.shape[0]
    
    def __getitem__(self, idx):
        path = self.studyuid[idx]
        path0 = self.studyuid[idx]
        # path = RETINAL_PATH + path+ ".jpg"  br01/BR_Snow01/pre_snow['img_img16266.jpg.pkl'].png  
        path = os.path.join(self.retinal_path, """img['img_"""+path+""".jpg.pkl'].png""")
        # /home/ubuntu/data/br_images/BR_Snow_orig/img['img_img16264.jpg.pkl'].png
        # print(path)
        image = cv2.imread(path)
      #  print(type(image))
        image = self.transform(image=image)['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensorV2()(image = image)["image"]
        labels = self.labels[idx]
        return image, labels, path0



# define datasets
class RANZCR_CLIP(Dataset):
    """RANZCR-CLIP Dataset Class

        Parameters
        ----------
        df_all : pandas DataFrame
            Contains all available labeled data (original train.csv format)
        df_studyIDs : pandas DataFrame
            lists all StudyInstanceUIDs to be used for this subset of RANZCR-CLIP
                (i.e. train IDs or val IDs)
        transform : transform types (i.e. what get_transform() returns)
    """
    def __init__(self, df_all, df_studyIDs, transform=None):
        df_subset = df_all[df_all["StudyInstanceUID"].isin(df_studyIDs[0])]
        self.studyuid = df_subset["StudyInstanceUID"].values
        self.labels = df_subset[LABELS].values
        self.transform = transform
        
    def __len__(self):
        return self.studyuid.shape[0]
    
    def __getitem__(self, idx):
        path = self.studyuid[idx]
        path = RANZCR_CLIP_PATH + "train/" + path + ".jpg"
        image = cv2.imread(path)
        image = self.transform(image=image)['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ToTensorV2()(image = image)["image"]
        labels = self.labels[idx]
        return image, labels

class MIMIC_CXR(Dataset):
    '''MIMIC-CXR Dataset Class

        Parameters
        ----------
        df : pandas DataFrame
            Contains all available labeled data (similar to the original train.csv format of RANZCR-CLIP)
            Except that the first column should be the jpg path of the image
        transform : transform types (i.e. what get_transform() returns)
    '''
    def __init__(self, df, transform=None):
        self.labels = df[LABELS].values
        self.transform = transform
        self.path = df["StudyInstanceUID"].values

    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):
        img_path = os.path.join(MIMIC_CXR_PATH, self.path[idx]).replace('dcm', 'jpg')
        image = cv2.imread(img_path)
        image = self.transform(image=image)['image']
        image = ToTensorV2()(image = image)["image"]
        labels = self.labels[idx]
        return image, labels

# define seeding function
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'Setting all seeds to be {seed} to reproduce...')

# define visualization functions

# define logging functions

# define misc functions
LABELS = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
        'Swan Ganz Catheter Present'
    ]

def readData(path):
    """Reads RANZCR-CLIP Train and Val data into DataFrames

    Parameters
    ----------
    path : str
        The path to the RANZCR-CLIP dataset

    Returns
    -------
    df_all pandas.DataFrame
        Contains all available labeled data (original RANZCR-CLIP train.csv format)
    df_train_val pandas.DataFrame
        Contains ALL StudyInstanceUIDs for both train and validation

    """
    print("Reading Data...")
    # df_all = pd.read_csv(os.path.join(path, "labelssubset.csv"))
    # df_train_val = pd.read_csv('train.txt', header=None)
    # print("Done!")
    # return df_all, df_train_val
    # df_all = pd.read_csv('/home/ubuntu/df1_train.csv')
    df_all = pd.read_csv(path)
    return df_all
def readTestData(path):
    """Reads RANZCR-CLIP Train and Val data into DataFrames
    Parameters
    ----------
    path : str
        The path to the RANZCR-CLIP dataset
    Returns
    -------
    df_all pandas.DataFrame
        Contains all available labeled data (original RANZCR-CLIP train.csv format)
    df_train_val pandas.DataFrame
        Contains ALL StudyInstanceUIDs for both train and validation
    """
    print("Reading Data...")
    # df_all = pd.read_csv(os.path.join(path, "labelssubset.csv"))
    # df_train_val = pd.read_csv('train.txt', header=None)
    # print("Done!")
    # return df_all, df_train_val
    # df_all = pd.read_csv('/home/ubuntu/df1_test.csv')
    df_all = pd.read_csv(path)
    return df_all

def k_fold_cross_val(df_train_val, df_all, k=3, stratified_grouped=False, val_perc=None):
    """Creates folds for cross validation or one split of specified percentage

    Parameters
    ----------
    df_train_val : pandas.DataFrame
        Contains ALL StudyInstanceUIDs for both train and validation
    k : int
        number of folds for cross validation (default is 3)
    stratified : boolean
        whether to preserve class distributions (default is False)
    grouped : boolean
        whether to control for patientIDs (no repeats across train and val sets)
            (default is False)
    val_perc : float
        fraction of total train_val set to use for validation (default is None)
            Only if you want to force a single iteration

    Returns
    -------
    folds : list
        list of length k with train and val StudyInstanceUIDs [[train1, val1], [train2, val2], ... ]
            Note: for non-NULL val_perc, folds is of length 1

    """
    folds = []
    if val_perc:
        train, val = train_test_split(df_train_val, test_size = val_perc, random_state=871)
        return [train, val]
    elif stratified_grouped:
        # encode each unique set of outcomes as a unique int
        enc = LabelEncoder()
        df_all['y'] = enc.fit_transform(df_all['patient_sex'])
        sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True)
        for train, val in sgkf.split(df_all['image_id'], df_all['y'], groups=df_all['patient_id']):
            train_ids = df_all['image_id'][train].to_frame()
            val_ids = df_all['image_id'][val].to_frame()
            train_ids = train_ids.rename(columns = {'image_id':0}).reset_index(drop=True)
            val_ids = val_ids.rename(columns = {'image_id':0}).reset_index(drop=True)
            # print(train_ids.head())
            folds.append([train_ids, val_ids])
        return folds
    else:
        kf = KFold(n_splits=k)
        for train_idx, val_idx in kf.split(df_train_val):
            # print(df_train_val.iloc[train_idx,:])
            folds.append([df_train_val.iloc[train_idx,:], df_train_val.iloc[val_idx,:]])
    return folds

def loadData(fold, df_all, batch_size, image_size):
    """Creates train and val loaders

    Parameters
    ----------
    fold : list of pandas.DataFrames
        [train, val]
    df_all : pandas.DataFrame
        Contains all available labeled data (original RANZCR-CLIP train.csv format)
    batch_size : int
        number of images per batch
    image_size : int
        size of images is image_size by image_size

    Returns
    -------
    train_loader : pytorch DataLoader
        loader for training set
    valid_loader : pytorch DataLoader
        loader for validation set

    """
    train_dataset = RANZCR_CLIP(df_all, fold[0], transform=get_transform(image_size, 'train'))
    train_loader = DataLoader(train_dataset, batch_size = batch_size , shuffle = True)    
    valid_dataset = RANZCR_CLIP(df_all, fold[1], transform=get_transform(image_size, 'val'))
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, valid_loader  

def loadTestData(df_test, df_all, batch_size, image_size):
    test_dataset = RANZCR_CLIP(df_all, df_test, transform=get_transform(image_size, 'val'))
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    return test_loader

def loadMIMIC(df, batch_size, image_size):
    MIMIC_dataset = MIMIC_CXR(df, transform=get_transform(image_size, 'train'))
    MIMIC_loader = DataLoader(MIMIC_dataset, batch_size = batch_size, shuffle = True)
    return MIMIC_loader 

def loadRetinalData(fold, df_all, batch_size, image_size):
    """Creates train and val loaders

    Parameters
    ----------
    fold : list of pandas.DataFrames
        [train, val]
    df_all : pandas.DataFrame
        Contains all available labeled data (original RANZCR-CLIP train.csv format)
    batch_size : int
        number of images per batch
    image_size : int
        size of images is image_size by image_size

    Returns
    -------
    train_loader : pytorch DataLoader
        loader for training set
    valid_loader : pytorch DataLoader
        loader for validation set

    """
    train_dataset = RETINAL(df_all, fold[0], transform=get_transform(image_size, 'train'))
    train_loader = DataLoader(train_dataset, batch_size = batch_size , shuffle = True)    
    valid_dataset = RETINAL(df_all, fold[1], transform=get_transform(image_size, 'val'))
    valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, valid_loader 


def loadRetinalData2(df_all, batch_size, image_size, retinal_path, class_column, channel_avg, channel_std, crop_dims, split='train', num_workers=0):
    """Creates train and val loaders

    Parameters
    ----------
    fold : list of pandas.DataFrames
        [train, val]
    df_all : pandas.DataFrame
        Contains all available labeled data (original RANZCR-CLIP train.csv format)
    batch_size : int
        number of images per batch
    image_size : int
        size of images is image_size by image_size

    Returns
    -------
    train_loader : pytorch DataLoader
        loader for training set
    valid_loader : pytorch DataLoader
        loader for validation set

    """
    train_dataset = RETINAL(df_all, retinal_path, class_column, transform=get_transform(image_size, channel_avg, channel_std, crop_dims, split=split))
    train_loader = DataLoader(train_dataset, batch_size = batch_size , shuffle = True, num_workers=num_workers)    
    # valid_dataset = RETINAL(df_all, fold[1], transform=get_transform(image_size, 'val'))
    # valid_loader = DataLoader(valid_dataset, batch_size = batch_size, shuffle = False)
    return train_loader

def loadTestRetinalData(df_test, df_all, batch_size, image_size):
    test_dataset = RETINAL(df_all, df_test, transform=get_transform(image_size, 'val'))
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    return test_loader



# Transform

def get_transform(image_size, channel_avg, channel_std, crop_dims, split = 'train'):
    transforms_train = albumentations.Compose([
       # albumentations.Crop(x_min=crop_dims[0], y_min=crop_dims[1], x_max=crop_dims[2], y_max=crop_dims[3], always_apply=True),
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(mean=channel_avg, std=channel_std, always_apply=True, max_pixel_value=255.0),
        albumentations.augmentations.dropout.cutout.Cutout (num_holes=58, max_h_size=16, max_w_size=16, fill_value=0, always_apply=False, p=0.25)
        #albumentations.Normalize(always_apply=True),
        #albumentations.HorizontalFlip(p=0.5),
        #albumentations.RandomBrightnessContrast(p=0.75),

        #albumentations.OneOf([
        #    albumentations.OpticalDistortion(distort_limit=1.),
        #    albumentations.GridDistortion(num_steps=5, distort_limit=1.),
        #], p=0.75),

        #albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
        #albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, border_mode=0, p=0.75),
     #   CutoutV2(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.75),
    ])
    transforms_val = albumentations.Compose([
       # albumentations.Crop(x_min=crop_dims[0], y_min=crop_dims[1], x_max=crop_dims[2], y_max=crop_dims[3], always_apply=True),
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize(mean=channel_avg, std=channel_std, always_apply=True, max_pixel_value=255.0),
    ])
    if split == 'train':
        return transforms_train
    elif split == 'val':
        return transforms_val
    else:
        raise NotImplementedError

def get_optim(model, optimizer, lr):
    if optimizer == 'Adam':
        optim = Adam(model.parameters(), lr=lr,weight_decay=lr/10)
    elif optimizer == 'SGD':
        optim = SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise NotImplementedError
    
    return optim

def get_lossfn(loss,weights):
    if loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif loss == 'FocalLoss':
        criterion = FocalLoss(gamma=2, alpha=weights)
    else:
        raise NotImplementedError
    return criterion

def read_json(config_file):
    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    return config

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum() 

if __name__ == "__main__":
    # train k networks reading from a config file for parameters
    config = read_json('./dsconfig.json')
    main(config)
