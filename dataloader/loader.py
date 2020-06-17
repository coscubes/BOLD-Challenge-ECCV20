import numpy as np
import csv
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import skvideo.io
import random

class BOLDTrainLoader(Dataset):
    def __init__(self, dataroot = None, input_size = 32):
        super().__init__()
        # We have modified the orginal annotations to discard the 
        # unrequired data and restructure it.
        # Minimum frame size = 10
        # Max frame size = 297
        # Num vids with less than 32 frames = 1168
        self.dataroot       = dataroot
        self.input_size     = input_size

        # Read data from CSV
        reader      = csv.reader(open(self.dataroot + "annotations_modified/train.csv", "r"), 
                                delimiter=",")
        self.data   = list(reader)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Separate the self.data row its components
        path        = self.data[index][0]
        vid_start   = int(self.data[index][-2])
        vid_end     = int(self.data[index][-1])
        emotions    = np.array(self.data[index][1:-2], dtype=np.float)


        # Read the video using scikit-video library. Takes a lot of time :(
        vid_array = None
        try:
            vid_array   = skvideo.io.vread(self.dataroot + "videos/" + path, 
                                        num_frames = vid_end)
        except FileNotFoundError:
            print(path)
        vid_array   = vid_array[vid_start:]
        joints      = np.load(self.dataroot + "joints/" + path[:-4] + ".npy")
        joints      = joints[vid_start:vid_end]

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

class BOLDValLoader(Dataset):
    def __init__(self, dataroot = None, input_size = 32):
        super().__init__()
        # We have modified the orginal annotations to discard the 
        # unrequired data and restructure it.
        # Minimum frame size = 10
        # Max frame size = 297
        # Num vids with less than 32 frames = 1168
        self.dataroot       = dataroot
        self.input_size     = input_size

        # Read data from CSV
        reader      = csv.reader(open(self.dataroot + "annotations_modified/val.csv", "r"), 
                                delimiter=",")
        self.data   = list(reader)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Separate the self.data row its components
        path        = self.data[index][0]
        vid_start   = int(self.data[index][-2])
        vid_end     = int(self.data[index][-1])
        emotions    = np.array(self.data[index][1:-2], dtype=np.float)
 
        # Read the video using scikit-video library. Takes a lot of time :(
        try:
            vid_array   = skvideo.io.vread(self.dataroot + "videos/" + path, 
                                        num_frames = vid_end)
        except FileNotFoundError:
            print(path)
        vid_array   = vid_array[vid_start:]
        joints      = np.load(self.dataroot + "joints/" + path[:-4] + ".npy")
        joints      = joints[vid_start:vid_end]

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
