import os

import torch

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_EPOCHS = 100
VAL_SIZE = 0.2
IMAGE_SIZE = 256
OUTPUT_DIR = "output"
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
TEST_SAMPLE_SIZE = 10


def ensure_output_dirs() -> None:
    for directory in [OUTPUT_DIR, MODELS_DIR, LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)
