# Rough work file to do code testing
# TODO Delete the file after the project is over.

from dataloader.loader import *
import config
import time

train_loader = BOLDTrainLoader(  dataroot = config.dataset_root, 
                                input_size = config.input_frames)

val_loader = BOLDValLoader(  dataroot = config.dataset_root, 
                                input_size = config.input_frames)
total_time = 0
count = 0
for i in range(len(train_loader)):
    now = time.time()
    try:
        x,y,z = train_loader[i]
    except FileNotFoundError:
        count += 1
    total_time += time.time() - now

print(count)
print(total_time)
print(total_time / len(train_loader))