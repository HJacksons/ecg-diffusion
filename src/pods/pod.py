from pods.diffwave import DiffWavePod
from pods.unet import UNetPod
from pods.steven import StevenPod
from pods.pulse2pulse import Pulse2PulsePod
import configuration as conf
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

        elif conf.MODEL == "steven":
            self.pod = StevenPod(lr=self.lr)

        elif conf.MODEL == "pulse2pulse":
            self.pod = Pulse2PulsePod(lr=self.lr)

        # Add any new network pods here

    def get_batch_size(self):
        return self.batch_size
