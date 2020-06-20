# Trash file to do testing
# Delete the file after the end of project

import  torch
import  torch.nn as nn
from    torch.optim import Adam
from    torch.utils.data import DataLoader
import  torchvision.models as models
from    torchvision.transforms.functional import *


import  config
from    dataloader.trainLoader import *
from    dataloader.valLoader import *


# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"

if config.device is None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = config.device

print("Using ", device, " for training")

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

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg11(pretrained=True).features
        print(self.vgg)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(7,7))
        self.fc1      = nn.Linear(25088, 4096)
        self.fc2      = nn.Linear(4096, 4096)
        self.fc3      = nn.Linear(4096, 29)
        self.act1     = nn.ReLU()
        self.act2     = nn.ReLU()
        self.dropout  = nn.Dropout()

    def forward(self, x):
        x = self.vgg(x)
        x = x.flatten(start_dim = 1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

net = model().to("cuda")

vid, joints, emo = train_data[2]
npimg = np.transpose(vid,(0, 3,1,2))
vid = torch.Tensor(npimg).to("cuda")
print(vid.shape)
y = net(vid)
print(y.shape)

optmizer    = Adam(net.parameters(), lr = config.learning_rate)
criterion   = torch.nn.MSELoss(reduction='sum')


for i in range(len(train_data)):
    vid, joints, emo    = train_data[i]
    vid                 = np.transpose(vid,(0, 3,1,2))
    vid                 = torch.Tensor(npimg).to(device)
    emo                 = torch.Tensor(np.ones((16,29)) * emo).to(device)

    pred = net(vid)
    loss = criterion(pred, emo)
    loss.backward()
    optmizer.step()
    print(loss)
