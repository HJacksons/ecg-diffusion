from src.pods.diffwave import DiffWavePod
from src.pods.unet import UNetPod
import src.configuration as conf
import wandb


class ModelPod:
    def __init__(self):
        self.lr = wandb.config.learning_rate if conf.USE_WEIGHTS_AND_BIASES else conf.HYPER_PARAMETERS['learning_rate']
        self.batch_size = wandb.config.batch_size if conf.USE_WEIGHTS_AND_BIASES else conf.HYPER_PARAMETERS['batch_size']

        # Unet configuration
        if conf.MODEL == "unet":
            self.pod = UNetPod(lr=self.lr)

        # DiffWave configuration
        elif conf.MODEL == "diffwave":
            self.pod = DiffWavePod(lr=self.lr)

    def get_batch_size(self):
        return self.batch_size
