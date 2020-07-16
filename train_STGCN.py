import  torch
import  torch.nn as nn
from    torch.optim import Adam
from    torch.utils.data import DataLoader
import  torchvision.models as models

import  config
from    dataloader.trainLoader_STGCN import *
from    dataloader.valLoader_STGCN import *

# Credits to https://github.com/yysijie/st-gcn for STGCN
from    models.STGCN import STGCN

import os
import time
import dill

# Set the right GPU on server  
if config.server:
    os.environ["CUDA_VISIBLE_DEVICES"]="3"


train_data = BOLDTrainLoader(
    dataroot    = config.dataset_root,
    input_size  = config.input_frames,
    height      = config.height,
    transform   = None
)


val_data =  BOLDValLoader(
    dataroot    = config.dataset_root,
    input_size  = config.input_frames,
    height      = config.height,
    transform   = None
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


model = STGCN(in_channels = 3, num_class = 29, layout='openpose', strategy='spatial',max_hop=1,
                 dilation=1,
                 edge_importance_weighting = True)

model.to(device)
optimizer   = Adam(model.parameters(), lr = config.learning_rate)
mse_loss   = torch.nn.MSELoss(reduction='sum')
ce_loss = nn.CrossEntropyLoss()


# Save an initial fully constructed model
#torch.save(model, config.model_path + "full_model.pt")


for epoch in range(config.num_epochs):
    total_train_loss = 0.0
    total_val_loss   = 0.0
    epoch_start = time.time()
    num_iter    = 0
    print("Starting epoch Num:", (epoch+1))
    for i, (joints, emotions) in enumerate(train_loader):
        optimizer.zero_grad()
        emotions= emotions.to(device)
        joints  = joints.to(device)
        preds   = model(joints)
        ind = emotions[:,:26].to(device).argmax(1)
        ce = ce_loss(preds[:,:26],ind)
        pad = torch.tanh(preds[:,26:])
        mse = mse_loss(pad,emotions[:,26:])
        loss    = ce + mse
        total_train_loss += loss
        loss.backward()
        optimizer.step()
        num_iter += 1

    print("Train Loss = ", total_train_loss / num_iter)

    num_iter = 0
    
    with torch.no_grad():
        for i, (joints, emotions) in enumerate(val_loader):
            optimizer.zero_grad()
            emotions= emotions.to(device)
            joints  = joints.to(device)
            preds   = model(joints)
            ind = emotions[:,:26].to(device).argmax(1)
            ce = ce_loss(preds[:,:26],ind)
            pad = torch.tanh(preds[:,26:])
            mse = mse_loss(pad,emotions[:,26:])
            loss    = ce + mse
            total_val_loss += loss
            num_iter += 1
    print("Validations Loss = ", total_val_loss / num_iter)
    print("Epoch time taken = ", time.time() - epoch_start)
    torch.save(model.state_dict(), config.model_path + "model-epoch-" + str(epoch) + ".pt")
