import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import config
from    models.I3D import InceptionI3d
from models.Inception_v4 import Inceptionv4

class Two_Stream(nn.Module):
    def __init__(self,class_num):
        super(Two_Stream,self).__init__()
        self.class_num = class_num
        self.rgb_stream = InceptionI3d(num_classes=400, in_channels=3,mode=2)
        self.rgb_stream.load_state_dict(torch.load("checkpoints/rgb_imagenet.pt"))
        self.rgb_stream.replace_logits(config.logits)
        self.joint_stream = Inceptionv4()
        self.linear_1 = nn.Linear(4608,1000)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(num_features=4608)
        self.linear_2 = nn.Linear(1000,class_num)
    
    def forward(self,frames,joints):
        frame_out = self.rgb_stream(frames)
        joints_out = self.joint_stream(joints)
        out  = torch.cat((frame_out,joints_out),dim=1)
        out = self.bn(out)
        out = self.relu(self.linear_1(out))
        out = self.linear_2(out)
        

        return out




