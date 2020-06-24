import  numpy as np
import  csv
import  cv2
from    torch.utils.data import Dataset
from    torchvision import transforms, utils
import  torch
import  skvideo.io
import  random
from    decord import VideoReader
from    decord import cpu, gpu
import  decord


class BOLDTestLoader(Dataset):
    def __init__(self, dataroot = None, input_size = 32,height = 256, transform=None,test_frames=5):
        super().__init__()
        # We have modified the orginal annotations to discard the 
        # unrequired data and restructure it.
        # Min num frames = 10
        # Max Num frames = 297
        # Num vids with less than 32 frames = 1168
        self.dataroot       = dataroot
        self.input_size     = input_size
        self.height         = height
        self.test_frames = test_frames
        decord.bridge.set_bridge('torch')

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
        
        try:
            vid_array     = self.get_video(self.dataroot + "videos/" + path)
            # vid_array   = skvideo.io.vread(self.dataroot + "videos/" + path) 
        except FileNotFoundError:
            print("FileNotFoundError", path)
            return

        end = min(len(joints), vid_end)
        vid_array   = vid_array[vid_start : end]
        joints      =  joints[vid_start : end]

        if vid_array.shape[0] < self.input_size:
            n = self.input_size // vid_array.shape[0] + 1
            vid_array = np.concatenate([vid_array] * n, axis = 0)
            joints    = np.concatenate([joints] * n, axis = 0)

        vid_array, joints = self.crop_video(vid_array, joints)
        vid_collec = []
        joints_collec = []
        for _ in range(self.test_frames):
            arr = random.sample(range(len(vid_array)), self.input_size)
            vid_collec.append(vid_array[arr])
            joints_collec.append(joints[arr])

        # print(vid_array.shape, joints.shape, emotions.shape)
        # if vid_array.shape[0] == 0 or joints.shape[0] == 0:
        #     print(vid_array.shape, joints.shape)
        #     print(path)
        vid_collec  = vid_collec.transpose([0,4,1,2,3])
        emotions   = np.array([emotions, emotions, emotions]).T
        return torch.Tensor(vid_collec).div(255.0), torch.Tensor(joints_collec), torch.Tensor(emotions)
    
    def get_video(self, fname):
        vid = []
        vr = VideoReader(fname, ctx=cpu(0))
        for i in range(len(vr)):
            vid.append(vr[i])
        return np.stack(vid)

    def crop_video(self, vid_array, joints):
        cropped_vid = []
        joint_vec   = []
        vid_height  = vid_array.shape[1]
        vid_width   = vid_array.shape[2]

        for i in range(vid_array.shape[0]):
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
                frame = cv2.resize(frame, (self.height, self.height))
                print("frame error")
            cropped_vid.append(frame)
            j_frame     = j_frame[2:].reshape(18, 3)
            j_frame     -= np.array([x, y, 0])
            j_frame     = j_frame.ravel()
            joint_vec.append(j_frame)

        cropped_vid = np.array(cropped_vid)
        joint_vec   = np.array(joint_vec)
        return cropped_vid, joint_vec


