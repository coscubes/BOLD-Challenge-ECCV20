import  torch
import  torch.nn as nn
from    torch.utils.data import DataLoader

import  config
from    dataloader.testLoader_Skepxles import *

import os
import time
import random
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

if config.server:
    os.environ["CUDA_VISIBLE_DEVICES"]="3"

test_data = BOLDTestLoader(
    dataroot    = config.dataset_root,
    input_size  = config.input_frames,
    height      = config.height,
    transform   = None,
    test_frames = config.test_frames
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
model.to(device)
model.eval()
loss = 0
mAP = 0
mRA = 0

with torch.no_grad():
    for i, (vid, skepxles, emotions) in enumerate(test_loader):
        if i%100 == 0:
            print(i)
        vid = vid.to(device)
        skepxles = skepxles.to(device)
        emotions= emotions.to(device)
        pred_avg = []
        for count in range(config.test_frames):
            vid_tensor = vid[:,count,:,:,:,:]
            skepxles_tensor = skepxles[:,count,:,:,:]
            pred = model(vid_tensor,skepxles_tensor)
            pred = pred.squeeze()
            pred_avg.append(pred)
            del vid_tensor
            del skepxles_tensor
            torch.cuda.empty_cache()
        pred_avg = torch.stack(pred_avg)
        pred_avg = torch.mean(pred_avg,dim=0)
        loss   += criterion(pred_avg, emotions)
        mAP += average_precision_score(emotions[:,:26], pred_avg[:,:26])
        mRA += roc_auc_score(emotions[:,:26], pred_avg[:,:26])

        del pred_avg
        torch.cuda.empty_cache()

print(loss,loss/i)
mR = loss/i
mRA = mRA/i
mAP = mAP/i

ERS = 0.5 * (mR + 0.5 * (mRA + mAP))
print("ERS : ",ERS)
print("mRA : ",mRA)
print("mAP : ",mAP)
print("mRR : ",mR)


