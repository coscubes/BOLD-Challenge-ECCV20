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
from    dataloader.trainLoader_Skepxles import *
from    dataloader.valLoader_Skepxles import *

# Credits to https://github.com/piergiaj/pytorch-i3d for I3D
from    models.Two_Stream import Two_Stream

import os
import time

# Set the right GPU on server  
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

mse_loss   = torch.nn.MSELoss(reduction='sum')
ce_loss = nn.CrossEntropyLoss()
model       = Two_Stream(class_num=29)
model.to(device)
optimizer   = Adam(model.parameters(), lr = config.learning_rate)


# Save an initial fully constructed model
torch.save(model, config.model_path + "full_model.pt")

for epoch in range(config.num_epochs):
    total_train_loss = 0.0
    total_val_loss   = 0.0
    epoch_start = time.time()
    num_iter    = 0
    print("Starting epoch Num:", (epoch+1))
    for i, (vid, joints, emotions) in enumerate(train_loader):
        optimizer.zero_grad()
        vid     = vid.to(device)
        emotions= emotions.to(device)
        joints  = joints.to(device)
        # torch.Size([8, 32, 224, 224, 3])
        preds   = model(vid,joints)
        ind = emotions[:,:26].to(device).argmax(1)
        ce = ce_loss(preds[:,:26],ind)
        mse = mse_loss(preds[:,26:],emotions[:,26:])
        loss    = ce + mse
        total_train_loss += loss
        loss.backward()
        optimizer.step()
        num_iter += 1

    print("Train Loss = ", total_train_loss / num_iter)

    num_iter = 0
    
    with torch.no_grad():
        for i, (vid, joints, emotions) in enumerate(val_loader):
            optimizer.zero_grad()
            vid     = vid.to(device)
            emotions= emotions.to(device)
            joints  = joints.to(device)
            preds   = model(vid,joints)
            ind = emotions[:,:26].to(device).argmax(1)
            ce = ce_loss(preds[:,:26],ind)
            mse = mse_loss(preds[:,26:],emotions[:,26:])
            loss    = ce + mse
            total_val_loss += loss
            num_iter += 1
    print("Validations Loss = ", total_val_loss / num_iter)
    print("Epoch time taken = ", time.time() - epoch_start)
    torch.save(model.state_dict(), config.model_path + "model-epoch-" + str(epoch) + ".pt")