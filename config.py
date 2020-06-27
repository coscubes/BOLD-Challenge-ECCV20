# Path to dataset (str)
# "/src/BOLD_public/"
# "/media/kaustubh/New_Volume/kaustubh_imp/datasets/BOLD_public/"
#dataset_root    = "/src/BOLD_public/"
dataset_root    = "/home/adityadhall/SS20/HLCV/Project/BOLD_public/"

# Learning Rate (float) <1.0
learning_rate   = 0.1

# Batch_size (int)
batch_size      = 1

# Number of epochs
num_epochs      = 40

# Number for frames for inputting the data (int)
input_frames    = 32

# Number of processes to launch
num_workers     = 0

# Shape of the input = height x height
height          = 224

# Set device automatically if None. or "cpu" / "cuda"
device          = None

# Number of outputs
logits          = 29

# TODO Sets GPU ID on the server
server          = False

# Checkpoints save path
model_path      = "checkpoints/I3D/"

# Checkpoint Index for testing
checkpoint_index = 39

# No. testing frames
test_frames = 3
