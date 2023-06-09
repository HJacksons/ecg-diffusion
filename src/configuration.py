from dotenv import load_dotenv
import torch
import os

# Load variables from .env file
load_dotenv()

# Paths
DATASET_PATH = os.path.abspath("./PTB-dataset")
PLOTS_FOLDER = os.path.abspath("./data/plots")
MODELS_FOLDER = os.path.abspath("./models")
GEN_MODELS_FOLDER = os.path.abspath("./models/generated")
GEN_DATA_FOLDER = os.path.abspath("./data/generated")

# General
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTION = "train"  # Options: ("train", "tune")
EPOCHS = 1000

# Network
MODEL = "pulse2pulse"  # Options: ("unet", "diffwave", "pulse2pulse", "steven")

# The feature that the Steven model should be trained for.
# Options: ("rr", "qrs", "pr", "qt", "VentricularRate", "R_Peak_i", "R_Peak_ii", "R_Peak_v1", "R_Peak_v2", "R_Peak_v3",
# "R_Peak_v4", "R_Peak_v5", "R_Peak_v6")
FEATURE = 'rr'

# Weights and Biases
WANDB_KEY = os.getenv("WANDB_KEY")
WANDB_ENTITY = os.getenv("WANDB_ENTITY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
USE_WEIGHTS_AND_BIASES = os.getenv("USE_WEIGHTS_AND_BIASES").lower() in ('true', '1')

# Train
HYPER_PARAMETERS = {
    'learning_rate': 0.0003,
    'batch_size': 32,
    'residual_layers': 36, 
    'residual_channels': 256, 
    'time_steps': 200,
}
