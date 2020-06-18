import numpy as np
import csv
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import skvideo.io
import random

class BOLDValLoader(Dataset):
    def __init__(self, dataroot = None, input_size = 32):
        super().__init__()
        # We have modified the orginal annotations to discard the 
        # unrequired data and restructure it.
        # Min num frames = 10
        # Max Num frames = 297
        # Num vids with less than 32 frames = 1168
        self.dataroot       = dataroot
        self.input_size     = input_size

        # Read data from CSV
        reader      = csv.reader(open(self.dataroot + "annotations_modified/val.csv", "r"), 
                                delimiter=",")
        self.data   = list(reader)
        rejected    = csv.reader(open(self.dataroot + "annotations_modified/val_rejected.csv", "r"), 
                                delimiter=",")
        rejected    = [i[0] for  i in list(rejected)]
        temp        = []
        
        for i in range(len(self.data)):
            if self.data[i][0] in rejected:
                continue
            else:
                temp.append(self.data[i])
        self.data   = temp

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Separate the self.data row its components
        path        = self.data[index][0]
        vid_start   = int(self.data[index][-2])
        vid_end     = int(self.data[index][-1])
        person_id   = int(self.data[index][-3])
        emotions    = np.array(self.data[index][1:-3], dtype=np.float)

        # Read the video using scikit-video library. Takes a lot of time :(
        vid_array = None
        try:
            vid_array   = skvideo.io.vread(self.dataroot + "videos/" + path) 
                                        # num_frames = vid_end)
        except FileNotFoundError:
            print(path)
            return
        vid_array   = vid_array[vid_start:]
        joints      = np.load(self.dataroot + "joints/" + path[:-4] + ".npy")
        # joints      = joints[vid_start:vid_end]
        print(vid_array.shape, joints.shape)
        return
        if (vid_end - vid_start) > self.input_size:
            # Randomly select frames from the given video of input_size
            arr         = random.sample(range(vid_end - vid_start), 
                                    self.input_size)
            arr.sort()
            vid_array   = vid_array[arr]
            # joints      = joints[arr]
        else:
            # Append the same video if the size is smaller than input_size
            while vid_array.shape[0] < self.input_size:
                vid_array = np.concatenate([vid_array, vid_array], axis = 0)
                joints    = np.concatenate([joints, joints], axis = 0)

        return vid_array, joints, emotions
