import os
import torch
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# General
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = os.path.abspath("../PTB-dataset")
PLOTS_FOLDER = "../data/plots"
ACTION = 'train'
BATCH_SIZE = 32
EPOCHS = 301

# Weights and Biases
WANDB_KEY = os.getenv("WANDB_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
USE_WEIGHTS_AND_BIASES = os.getenv("USE_WEIGHTS_AND_BIASES").lower() in ('true', '1')

# Train
HYPER_PARAMETERS = {
    'LEARNING_RATE': 0.0003,
}
