import numpy as np
import csv
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import skvideo.io
import random

class BOLDTrainLoader(Dataset):
    def __init__(self, dataroot = None, input_size = 32, height = 256, transform=None):
        super().__init__()
        # We have modified the orginal annotations to discard the 
        # unrequired data and restructure it.
        # Min num frames = 10
        # Max Num frames = 297
        # Num vids with less than 32 frames = 1168
        self.dataroot       = dataroot
        self.input_size     = input_size
        self.height         = height

        # Read data from CSV
        reader      = csv.reader(open(self.dataroot + "annotations_modified/train.csv", "r"), 
                                delimiter=",")
        self.data   = list(reader)
        rejected    = csv.reader(open(self.dataroot + "annotations_modified/train_rejected.csv", "r"), 
                                delimiter=",")
        rejected    = [i[0] for  i in list(rejected)]
        temp        = []
        
        for i in range(len(self.data)):
            if self.data[i][0] in rejected:
                continue
            else:
                temp.append(self.data[i])
        self.data   = temp
        self.transform = transform
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Separate the self.data row its components
        path        = self.data[index][0]
        vid_start   = int(self.data[index][-2])
        vid_end     = int(self.data[index][-1])
        person_id   = int(self.data[index][-3])
        emotions    = np.array(self.data[index][1:-3], dtype=np.float)
        joints      = np.load(self.dataroot + "joints/" + path[:-4] + ".npy")
        joints      = joints[np.where(joints[:,1] == person_id)] 
        
        # Read the video using scikit-video library. Takes a lot of time :(
        vid_array   = None
        try:
            vid_array   = skvideo.io.vread(self.dataroot + "videos/" + path) 
        except FileNotFoundError:
            print(path)
            return
        
        # I know it is redundant code but please keep it as it is.
        # Debugging it is hard
        if joints.shape[0] == vid_array.shape[0]:
            # Randomly select frames from the given video of input_size
            joints      = joints[vid_start : vid_end]
            vid_array   = vid_array[vid_start : vid_end]
            if vid_end - vid_start > self.input_size:
                # The ideal case where num frames is greater than input size
                arr         = random.sample(range(len(vid_array)), 
                                        self.input_size)
                arr.sort()
                vid_array   = vid_array[arr]
                
                joints      = joints[arr]
            else:
                # Append the same video if the size is smaller than input_size
                while vid_array.shape[0] < self.input_size:
                    vid_array = np.concatenate([vid_array, vid_array], axis = 0)
                    joints    = np.concatenate([joints, joints], axis = 0)
                
                arr         = random.sample(range(len(vid_array)), 
                                        self.input_size)
                arr.sort()
                vid_array   = vid_array[arr]
                
                joints      = joints[arr]
        else:
            if vid_array.shape[0] < joints.shape[0]:
                joints      = joints[:vid_array.shape[0]]
            else:
                vid_array   = vid_array[:joints.shape[0]]

            if vid_array.shape[0] > self.input_size:
                # The ideal case where num frames is greater than input size
                arr         = random.sample(range(len(vid_array)), 
                                        self.input_size)
                arr.sort()
                vid_array   = vid_array[arr]
                
                joints      = joints[arr]
            else:
                # Append the same video if the size is smaller than input_size
                while vid_array.shape[0] < self.input_size:
                    vid_array = np.concatenate([vid_array, vid_array], axis = 0)
                    joints    = np.concatenate([joints, joints], axis = 0)
                arr         = random.sample(range(len(vid_array)), 
                                        self.input_size)
                arr.sort()
                vid_array   = vid_array[arr]
                
                joints      = joints[arr]
        
        # print(joints.shape, vid_array.shape, vid_end - vid_start)
        # Crop the video to height x height
        cropped_vid = []
        joint_vec   = []
        vid_height  = vid_array.shape[1]
        vid_width   = vid_array.shape[2]

        for i in range(self.input_size):
            j_frame = joints[i]
            x,y,z   = np.mean(j_frame[2:].reshape(18, 3), axis=0)
            
            left    = x - (self.height / 2)
            top     = y - (self.height / 2)
            right   = x + (self.height / 2)
            bottom  = y + (self.height / 2)
            if top < 0:
                top     = 0
                bottom  = top + self.height
            if bottom > vid_height:
                bottom = vid_height
                top    = bottom - self.height
            if left < 0:
                left    = 0
                right   = left + self.height                
            if right > vid_width:
                right = vid_width
                left  = vid_width - self.height

            frame = vid_array[i][ int(top):int(bottom), int(left):int(right)]
            if frame.shape != (224, 224, 3):
                print(vid_height, vid_width, x, y, top, bottom, left, right, frame.shape)
            cropped_vid.append(frame)
            j_frame     = j_frame[2:].reshape(18, 3)
            j_frame     -= np.array([x, y, 0])
            j_frame     = j_frame.ravel()
            joint_vec.append(j_frame)

        cropped_vid = np.array(cropped_vid)
        joint_vec   = np.array(joint_vec)
        # print(cropped_vid.shape, joint_vec.shape)
        if self.transform:
            cropped_vid = self.transform(cropped_vid)
        return cropped_vid, joint_vec, emotions


