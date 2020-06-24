import  torch
from    torch.utils.data import DataLoader

import  config
from    dataloader.testLoader import *

import os
import time

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

model.eval()

for i, (vid, joints, emotions) in enumerate(test_loader):
    print(type(vid))
    vid_array = vid.cpu().detach().numpy()
    joints_array = joints.cpu().detach().numpy()
    emotions_array = emotions.cpu().detach().numpy()
    print(vid_array.shape,joints_array.shape)
    break