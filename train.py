import  torch
import  torch.nn as nn
from    torch.optim import Adam
from    torch.utils.data import DataLoader
import  torchvision.models as models
from    torchvision.transforms.functional import *


import  config
from    dataloader.trainLoader import *
from    dataloader.valLoader import *
from    models.I3D import InceptionI3d

# Set the right GPU on server
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

if config.device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = config.device

print("Using ", device, " for training")

optmizer    = Adam(net.parameters(), lr = config.learning_rate)
criterion   = torch.nn.MSELoss(reduction='sum')
model       = InceptionI3d(num_classes=400, in_channels=3)
model.replace_logits(config.logits)
