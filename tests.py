# Rough work file to do code testing
# TODO Delete the file after the project is over.

from dataloader.trainLoader import *
from dataloader.valLoader import *
import config
import time


train_loader = BOLDTrainLoader(  
    dataroot    = config.dataset_root, 
    input_size  = config.input_frames,
    height      = config.height
)

val_loader = BOLDValLoader(
    dataroot    = config.dataset_root, 
    input_size  = config.input_frames,
    height      = config.height
)

total_time = 0
for i in range(len(val_loader)):
    now = time.time()
    val_loader[i]
    total_time += time.time() - now

print(total_time)
print(total_time / len(val_loader))
