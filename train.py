import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.trainLoader import *
from dataloader.valLoader import *
import config

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

train_data = BOLDTrainLoader(
    dataroot    = config.dataset_root,
    input_size  = config.input_frames,
    height      = config.height
)

val_data =  BOLDValLoader(
    dataroot    = config.dataset_root,
    input_size  = config.input_frames,
    height      = config.height
)

train_loader = DataLoader(
    train_data,
    batch_size  = config.batch_size,
    shuffle     = True,
    num_workers = config.num_workers
)

val_loader = DataLoader(
    val_data,
    batch_size  = config.batch_size,
    shuffle     = False,
    num_workers = config.num_workers
)