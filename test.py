import  torch
import  torch.nn as nn
from    torch.utils.data import DataLoader

import  config
from    dataloader.testLoader import *

import os
import time
import random
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import r2_score
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
emotion_record = []
pred_record = []
mAP = 0
mRA = 0
with torch.no_grad():
    for i, (vid, joints, emotions) in enumerate(test_loader):
        if i%100 == 0:
            print(i)
        vid = vid.to(device)
        joints = joints.to(device)
        emotions= emotions.to(device)
        emotions = emotions.squeeze()
        pred_avg = []
        for count in range(config.test_frames):
            vid_tensor = vid[:,count,:,:,:,:]
            joints_tensor = joints[:,count,:]
            pred = model(vid_tensor)
            pred = pred.squeeze()
            pred_avg.append(pred)
            del vid_tensor
            del joints_tensor
            torch.cuda.empty_cache()
        pred_avg = torch.stack(pred_avg)
        pred_avg = torch.mean(pred_avg,dim=0)
        emotions = emotions.detach().cpu().numpy()
        pred_avg = pred_avg.detach().cpu().numpy()
        emotions_OH = np.zeros_like(emotions[:26])
        emotions_OH[np.argmax(emotions[:26])] = 1
        pred_OH = np.zeros_like(pred_avg[:26])
        pred_OH[np.argmax(pred_avg[:26])] = 1
        emotion_record.append(emotions[26:].tolist())
        pred_record.append(pred_avg[26:].tolist())
        mAP += average_precision_score(emotions_OH, pred_OH)
        mRA += roc_auc_score(emotions_OH, pred_OH)
        

        del pred_avg
        torch.cuda.empty_cache()


mR = r2_score(np.array(emotion_record),np.array(pred_record)) 
mRA = mRA/i
mAP = mAP/i

ERS = 0.5 * (mR + 0.5 * (mRA + mAP))
print("ERS : ",ERS)
print("mRA : ",mRA)
print("mAP : ",mAP)
print("mRR : ",mR)

