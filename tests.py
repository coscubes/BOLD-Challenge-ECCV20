# Trash file to do testing
# Delete the file after the end of project

import  torch
import  torch.nn as nn
from    torch.optim import Adam
from    torch.utils.data import DataLoader
import  torchvision.models as models
from    torchvision.transforms import *

from    torchvideotransforms.video_transforms import *
from    torchvideotransforms.volume_transforms import *
from    torchvideotransforms.stack_transforms import *

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

def train():
    class model(nn.Module):
        def __init__(self):
            super().__init__()
            self.vgg = models.vgg11(pretrained=True).features
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

    optmizer    = Adam(net.parameters(), lr = config.learning_rate)
    criterion   = torch.nn.MSELoss(reduction='sum')

    print("we reached here")
    for i, (vid, joints, emo) in enumerate(train_loader):
        # vid, joints, emo    = train_data[i]

        print(vid.shape, joints.shape, emo.shape)
        # vid                 = np.transpose(vid,(0, 3,1,2))
        # vid                 = vid.to(device)
        # print(vid.shape)
        # emo                 = torch.Tensor(np.ones((16,29)) * emo).to(device)

        # pred = net(vid)
        # loss = criterion(pred, emo)
        # loss.backward()
        # optmizer.step()
        # print(loss)

if __name__ == '__main__':
    train()
