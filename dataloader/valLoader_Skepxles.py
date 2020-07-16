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

class BOLDValLoader(Dataset):
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
        self.skep_thresh    = skep_thresh
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
        
        arr         = random.sample(range(len(vid_array)), self.input_size)
        vid_array   = vid_array[arr]
        joints      = joints[arr]
        vid_array = self.crop_video(vid_array,joints)
        skepxles = self.compute_skepxles(joints[:,2:])
        
        # print(vid_array.shape, joints.shape, emotions.shape)
        # if vid_array.shape[0] == 0 or joints.shape[0] == 0:
        #     print(vid_array.shape, joints.shape)
        #     print(path)
        vid_array  = vid_array.transpose([3,0,1,2])
        skepxles = skepxles.transpose([2,0,1])
        return torch.Tensor(vid_array).div(255.0), torch.Tensor(skepxles), torch.Tensor(emotions)
    

    def transform_joints(self,joint):
        joint_len = joint.shape[0]
        ret = np.zeros((joint_len//3,3))
        for i in range(joint_len//3):
            ret[i,:] = joint[i*3:i*3 + 3]
        return ret

    def check_scatter(self,collec):
        gamma = 0
        for mat in range(self.input_size):
            for coeff in range(16):
                x,y = np.where(collec[mat] == coeff)
                for mat_prime in range(self.input_size):
                    if mat == mat_prime:
                        continue
                    x_prime,y_prime = np.where(collec[mat_prime] == coeff)
                    gamma += max(abs(x[0] - x_prime[0]),abs(y[0] - y_prime[0]))
        if gamma > self.skep_thresh:
            return True
        else:
            return False
                

    def create_skepxle_coeff(self):
        while(True):
            collec = []
            for i in range(self.input_size):
                coeff = np.arange(16).reshape((4,4))
                np.random.shuffle(coeff)
                collec.append(coeff)
            ret = self.check_scatter(collec)
            if ret == True:
                break
        return collec
            
    
    def compute_skepxles(self,joints):
        skepxles = np.zeros((4*self.input_size,4*self.input_size,6))
        joint_prev_frame = None
        for frame in range(self.input_size):
            joint = joints[frame,:]
            joint = self.transform_joints(joint)
            '''
            joint_ignore = []
            for i in range(joint.shape[0]):
                if (joint[i,:] == np.array([0.0,0.0,0.0])).all():
                    joint_ignore.append(i)
            random.shuffle(joint_ignore)
            if len(joint_ignore) > 2:
                joint[joint_ignore[0],:] = joint[-1,:]
                joint[joint_ignore[1],:] = joint[-2,:]
                joint = joint[:-2,:]#Selected 16 joints eliminating one joint randomly with (0,0,0) coordinate
            else:
                joint = joint[:-2,:]
            '''
            joint[10,:] = joint[-1,:]
            joint[12,:] = joint[-2,:]
            joint = joint[:-2,:] # leaving both ankel positions 
            collec = self.create_skepxle_coeff()

            for i,coeff in enumerate(collec):
                temp = joint[coeff,:]

                skepxles[i*4 : i*4 + 4,frame*4:frame*4 + 4,0:3] = temp
                if frame != 0:
                    skepxles[i*4 : i*4 + 4,frame*4:frame*4 + 4,3:] = temp - joint_prev_frame[coeff,:]


            joint_prev_frame = joint.copy()        
        return skepxles

        


    def get_video(self, fname):
        vid = []
        vr = VideoReader(fname, ctx=cpu(0))
        for i in range(len(vr)):
            vid.append(vr[i])
        return np.stack(vid)

    def crop_video(self, vid_array,joints):
        cropped_vid = []
        #joint_vec   = []
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
                frame = cv2.resize(frame, (self.height, self.height))
                print("frame error")
            cropped_vid.append(frame)
            '''
            j_frame     = j_frame[2:].reshape(18, 3)
            j_frame     -= np.array([x, y, 0])
            j_frame     = j_frame.ravel()
            joint_vec.append(j_frame)
            '''
        cropped_vid = np.array(cropped_vid)
        #joint_vec   = np.array(joint_vec)
        return cropped_vid
        