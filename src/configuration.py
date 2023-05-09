from dotenv import load_dotenv
import torch
import os

# Load variables from .env file
load_dotenv()

# General
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = os.path.abspath("../PTB-dataset")
PLOTS_FOLDER = os.path.abspath("../data/plots")
MODELS_FOLDER = os.path.abspath("../models")
ACTION = "train"  # Options: ("train", "tune")
EPOCHS = 1000

# Network
MODEL = "diffwave"  # Options: ("diffwave", "unet")

# Validation
TRAINED_MODEL_PATH = os.path.abspath("../models/Steven_rr.pt")
VALIDATE_DIFFUSION = os.getenv("VALIDATE_DIFFUSION").lower() in ('true', '1')
VALIDATION_SAMPLES = 100

# Weights and Biases
WANDB_KEY = os.getenv("WANDB_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
USE_WEIGHTS_AND_BIASES = os.getenv("USE_WEIGHTS_AND_BIASES").lower() in ('true', '1')

# Train
HYPER_PARAMETERS = {
    'learning_rate': 0.0002,
    'batch_size': 16,
    'residual_layers': 36, 
    'residual_channels': 256, 
}
