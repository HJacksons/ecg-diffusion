import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# General
PLOTS_FOLDER = "../data/plots"
ACTION = 'train'

# Weights and Biases
WANDB_KEY = os.getenv("WANDB_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
USE_WEIGHTS_AND_BIASES = os.getenv("USE_WEIGHTS_AND_BIASES")

# Train
TRAIN_CONFIGURATION = {
    'DATASET_OPTION': '../PTB-dataset',
    'LEARNING_RATE': 0.0003,
    'BATCH_SIZE': 32,
    'EPOCHS': 301,
}
