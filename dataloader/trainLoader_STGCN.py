import  numpy as np
import  csv
import  cv2
from    torch.utils.data import Dataset
from    torchvision import transforms, utils
import  torch
import  skvideo.io
import  random
from    torchvision.transforms.functional import to_pil_image
from    decord import VideoReader
from    decord import cpu, gpu
import  decord

class BOLDTrainLoader(Dataset):
    def __init__(self, dataroot = None, input_size = 32, height = 256, transform=None,skep_thresh = 10):
        super().__init__()
        # We have modified the orginal annotations to discard the 
        # unrequired data and restructure it.
        # Min num frames = 10
        # Max Num frames = 297
        # Num vids with less than 32 frames = 1168
        self.dataroot       = dataroot
        self.input_size     = input_size
        self.height         = height
        self.skep_thresh = skep_thresh
        decord.bridge.set_bridge('torch')

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

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        path        = self.data[index][0]
        vid_start   = int(self.data[index][-2])
        vid_end     = int(self.data[index][-1])
        person_id   = int(round(float(self.data[index][-3])))
        emotions    = np.array(self.data[index][1:-3], dtype=np.float)
        joints      = np.load(self.dataroot + "joints/" + path[:-4] + ".npy")
        joints      = joints[np.where(joints[:,1] == person_id)]
        vid_array   = None
      
        end = min(len(joints), vid_end)
        joints      =  joints[vid_start : end]
        
        if joints.shape[0] < self.input_size:
            n = self.input_size // joints.shape[0] + 1
            joints    = np.concatenate([joints] * n, axis = 0)
      
        arr         = random.sample(range(len(joints)), self.input_size)
        joints      = joints[arr]
        joints_clean = self.clean_joints(joints[:,2:])
        joints_clean = np.expand_dims(joints_clean,axis = 0)
        joints_clean  = joints_clean.transpose([3,1,2,0])
        return torch.Tensor(joints_clean), torch.Tensor(emotions)
    

    def transform_joints(self,joint):
        joint_len = joint.shape[0]
        ret = np.zeros((joint_len//3,3))
        for i in range(joint_len//3):
            ret[i,:] = joint[i*3:i*3 + 3]
        return ret
    
    def clean_joints(self,joints):
        ret = np.zeros((joints.shape[0],18,3))
        for i in range(joints.shape[0]):
            temp = self.transform_joints(joints[i,:])
            for j in range(18):
                if temp[j,2] == 0.0:
                    temp[j,0] = 0.0
                    temp[j,1] = 0.0
            ret[i,:] = temp
        return ret