import  torch
import  torch.nn as nn
from    torch.utils.data import DataLoader

import  config
from    dataloader.testLoader import *

import os
import time
import random

if config.server:
    os.environ["CUDA_VISIBLE_DEVICES"]="3"

test_data = BOLDTestLoader(
    dataroot    = config.dataset_root,
    input_size  = config.input_frames,
    height      = config.height,
    transform   = None
)

test_loader = DataLoader(
    dataset     = test_data,
    batch_size  = 1,
    shuffle     = False,
    num_workers = config.num_workers
)

if config.device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = config.device

print("Using ", device, " for testing")

model = torch.load(config.model_path + "full_model.pt")
model.load_state_dict(torch.load(config.model_path + "model-epoch-" + str(config.checkpoint_index) + ".pt"))
criterion   = torch.nn.MSELoss(reduction='sum')
model.eval()
loss = 0
for i, (vid, joints, emotions) in enumerate(test_loader):
    vid_array = vid.cpu().detach().numpy()
    joints_array = joints.cpu().detach().numpy()
    #emotions_array = emotions.cpu().detach().numpy()
    emotions= emotions.to(device)
    emotions = emotions.squeeze()
    pred_avg = []
    for count in range(config.test_frames):
        arr = random.sample(range(vid_array.shape[2]), config.input_frames)
        vid_tensor = torch.from_numpy(vid_array[:,:,arr,:,:]).to(device)
        joints_tensor = torch.from_numpy(joints_array[:,arr,:]).to(device)
        pred = model(vid_tensor)
        pred = pred.squeeze()
        pred_avg.append(pred)
    pred_avg = torch.stack(pred_avg)
    pred_avg = torch.mean(pred_avg,dim=0)
    loss   += criterion(pred_avg, emotions)

print(loss,loss/i)