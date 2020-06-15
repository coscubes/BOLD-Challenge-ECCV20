import numpy as np
import csv
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class BOLDTrainLoader(Dataset, dataroot = None):
    def __init__(self):
        super().__init__()
        
        # Read the training files
        self.dataroot = dataroot + "train.csv"
        reader = csv.reader(open(self.dataroot, "r"), delimiter=",")
        self.data = list(reader)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
