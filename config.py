# Path to dataset (str)
dataset_root    = "/media/kaustubh/New_Volume/kaustubh_imp/datasets/BOLD_public/"

# Learning Rate (float) <1.0
learning_rate   = 0.1

# Batch_size (int)
batch_size      = 4

# Number of epochs
num_epochs      = 10

# Number for frames for inputting the data (int)
input_frames    = 16

# Number of processes to launch
num_workers     = 10

# Shape of the input = height x height
height          = 224

# Set device automatically if None. or "cpu" / "cuda"
device          = None

# Number of outputs
logits          = 29