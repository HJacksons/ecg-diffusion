from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from datasets import PTB_Dataset, TensorDataset
from matplotlib import pyplot as plt
import configuration as conf
from pathlib import Path
from typing import Tuple
import numpy as np
import random
import wandb
import torch


def init_wandb(api_key, project: str, config: dict) -> wandb:
    wandb.login(key=api_key)
    wandb.init(project=project, entity=conf.WANDB_ENTITY, config=config)
    return wandb


def create_folder_if_not_exists(folder: str):
    if not (Path(folder).is_dir()):
        Path(folder).mkdir(parents=True, exist_ok=True)


def fix_seed():
    random_seed = 123
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_and_save_plot(generated_leads_two_eight, filename, file_extension='.png'):
    line_width = 0.3
    fig, axs = plt.subplots(4, 2, figsize=(18, 12))

    for i in range(4):
        for j in range(2):
            axs[i, j].yaxis.set_major_locator(MultipleLocator(0.5))
            axs[i, j].yaxis.set_minor_locator(AutoMinorLocator(4))
            axs[i, j].grid(which='major', color='#CCCCCC', linestyle='--', linewidth=line_width)
            axs[i, j].grid(which='minor', color='#CCCCCC', linestyle=':', linewidth=line_width / 2)

    for i in range(3):
        axs[i + 1, 0].plot(generated_leads_two_eight[i], linewidth=line_width)
    for i in range(4):
        axs[i, 1].plot(generated_leads_two_eight[i + 3], linewidth=line_width)
    fig.savefig(Path(filename + file_extension))
    plt.close(fig)


def get_dataloader(
        target="train",
        batch_size=conf.TRAIN_CONFIGURATION['BATCH_SIZE'],
        shuffle=True
) -> Tuple[TensorDataset, torch.utils.data.DataLoader]:
    dataset = PTB_Dataset(data_dirs=conf.TRAIN_CONFIGURATION['DATASET_OPTION'], target=target)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        drop_last=True
    )
    return dataset, dataloader
