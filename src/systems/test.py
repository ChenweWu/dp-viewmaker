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
from src.models import resnet_var
import torch_dct as dct
import pytorch_lightning as pl
# import wandb
import timm
# model = resnet_var.resnet200D_v( num_classes=2)

# model.load_state_dict(torch.load('/home/ubuntu/attacker_xray.pth')) 
# model = models.Resnet200D(model_name='resnet200d', number_of_classes=5)
# model.load_state_dict(torch.load('/home/ubuntu/Xray/results/2023-12-07_02-22-15/best_checkpoints/checkpoint_2.pt')['state_dict'])

model = timm.create_model('resnet200d', pretrained=True)
model.fc = nn.Linear(model.fc.in_features, out_features=5)
model.load_state_dict(torch.load('/home/ubuntu/Xray/results/2023-12-07_02-22-15/best_checkpoints/checkpoint_2.pt')['state_dict'])