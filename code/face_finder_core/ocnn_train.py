"""
Fine-tunes a pretrained ResNet18/ResNet50 model using the VGGFace2 Dataset and the ArcFace learning method
"""

import json
import logging
import os
from pathlib import Path

import albumentations as A

# Preprocess
import cv2
import faiss
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as torchvision
from albumentations.pytorch import ToTensorV2

# Face Detection
from facenet_pytorch import MTCNN
from PIL import Image

# ArcFace loss(losses.ArcFaceLoss)
from pytorch_metric_learning import losses
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Visulizations/Others
from tqdm import tqdm

# Validate directory structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "face_recognition_training"
VAL_DIR = PROJECT_ROOT / "data" / "face_recognition_validation"
TEST_DIR = PROJECT_ROOT / "data" / "face_recognition_test"
MIN_IDENTITIES = 10
train_path = Path(TRAIN_DIR)
val_path = Path(VAL_DIR)
test_path = Path(TEST_DIR)
sum_subdirs = (
    sum(1 for x in train_path.iterdir() if x.is_dir())
    + sum(1 for x in val_path.iterdir() if x.is_dir())
    + sum(1 for x in test_path.iterdir() if x.is_dir())
)

if sum_subdirs < MIN_IDENTITIES:
    raise RuntimeError("own_cnn: not enough identities detected!")
