import  torch
import  torch.nn as nn
from    torch.optim import Adam
from    torch.utils.data import DataLoader
import  torchvision.models as models

# Credits to https://github.com/hassony2/torch_videovision for this
from    torchvideotransforms.video_transforms import *
from    torchvideotransforms.volume_transforms import *
from    torchvideotransforms.stack_transforms import *

import  config
from    dataloader.trainLoader import *
from    dataloader.valLoader import *

# Credits to https://github.com/piergiaj/pytorch-i3d for I3D
from    models.I3D import InceptionI3d

# Set the right GPU on server
import os

if config.server:
    os.environ["CUDA_VISIBLE_DEVICES"]="3"

train_transforms    = Compose([
    ColorJitter(0.5, 0.5, 0.25),
    ClipToTensor(channel_nb=3),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms      = Compose([
    ClipToTensor(channel_nb=3),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = BOLDTrainLoader(
    dataroot    = config.dataset_root,
    input_size  = config.input_frames,
    height      = config.height,
    transform   = train_transforms
)

val_data =  BOLDValLoader(
    dataroot    = config.dataset_root,
    input_size  = config.input_frames,
    height      = config.height,
    transform   = val_transforms
)

train_loader = DataLoader(
    dataset     = train_data,
    batch_size  = config.batch_size,
    shuffle     = True,
    num_workers = config.num_workers
)

val_loader = DataLoader(
    dataset     = val_data,
    batch_size  = config.batch_size,
    shuffle     = False,
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

criterion   = torch.nn.MSELoss(reduction='sum')
model       = InceptionI3d(num_classes=400, in_channels=3)

model.load_state_dict(torch.load("checkpoints/rgb_imagenet.pt"))
model.replace_logits(config.logits)
model.to(device)
optimizer   = Adam(model.parameters(), lr = config.learning_rate)

for epoch in range(config.num_epochs):
    print("Starting epoch Num:", (epoch+1))
    for i, (vid, joints, emotions) in enumerate(train_loader):
        optimizer.zero_grad()
        vid     = vid.to(device)
        emotions= emotions.to(device)
        joints  = joints.to(device)
        # torch.Size([8, 32, 224, 224, 3])
        preds   = model(vid)
        loss    = criterion(preds, emotions)
        print(loss)
        loss.backward()
        optimizer.step()
