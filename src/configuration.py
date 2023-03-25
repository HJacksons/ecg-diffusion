import os
import torch
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# General
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = os.path.abspath("../PTB-dataset")
PLOTS_FOLDER = os.path.abspath("../data/plots")
MODELS_FOLDER = os.path.abspath("../data/trained_models")
ACTION = "train"  # Options: ("train", "tune")

# Validation
TRAINED_MODEL_PATH = os.path.abspath("rr_prediction_model.pt")
VALIDATE_DIFFUSION = os.getenv("VALIDATE_DIFFUSION").lower() in ('true', '1')
VALIDATION_SAMPLES = 100

# Weights and Biases
WANDB_KEY = os.getenv("WANDB_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
USE_WEIGHTS_AND_BIASES = os.getenv("USE_WEIGHTS_AND_BIASES").lower() in ('true', '1')

# Train
HYPER_PARAMETERS = {
    'learning_rate': 0.0003,
    'batch_size': 32,
    'epochs': 301,
}
