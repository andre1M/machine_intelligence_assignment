import torch


SEED = 11
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 200
LR = 1e-2
MOMENTUM = 0.9
WD = 1e-4
BATCH_SIZE = 128
COMP_RATE = 0.8
OUTPUT_DIR = '../output'
