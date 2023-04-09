from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from datasets import PTB_Dataset, TensorDataset
from matplotlib import pyplot as plt
from src.networks.Steven import KanResWide_X
import configuration as conf
from pathlib import Path
from typing import Tuple
import numpy as np
import random
import wandb
import torch


def init_wandb() -> wandb:
    wandb.login(key=conf.WANDB_KEY)
    wandb.init(
        project=conf.WANDB_PROJECT,
        entity=conf.WANDB_ENTITY,
        config=conf.HYPER_PARAMETERS,
        dir='../data'
    )
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


def create_and_save_plot(generated_leads_I_VIII, filename, file_extension='.png', label=None):
    LINE_WIDTH = 0.3
    fig, axs = plt.subplots(4, 2, figsize=(18, 12))

    for i in range(4):
        for j in range(2):
            axs[i, j].yaxis.set_major_locator(MultipleLocator(0.5))
            axs[i, j].yaxis.set_minor_locator(AutoMinorLocator(4))
            axs[i, j].grid(which='major', color='#CCCCCC', linestyle='--', linewidth=LINE_WIDTH)
            axs[i, j].grid(which='minor', color='#CCCCCC', linestyle=':', linewidth=LINE_WIDTH / 2)

    for i in range(4):
        # axs[i, 0].plot(leadsI_VIII[i], linewidth=LINE_WIDTH)
        axs[i, 0].plot(generated_leads_I_VIII[i], linewidth=LINE_WIDTH)
    for i in range(4):
        # axs[i, 1].plot(leadsI_VIII[i+3], linewidth=LINE_WIDTH)
        axs[i, 1].plot(generated_leads_I_VIII[i + 3], linewidth=LINE_WIDTH)
    
    if label is not None:
        fig.text(0.5, 0.95, f'RR interval: {label}', ha='center', fontsize=16)

    fig.savefig(Path(filename + file_extension))
    plt.close(fig)


def get_dataloader(
        target="train",
        batch_size=conf.HYPER_PARAMETERS['batch_size'],
        shuffle=True
) -> Tuple[TensorDataset, torch.utils.data.DataLoader]:
    dataset = PTB_Dataset(data_dirs=conf.DATASET_PATH, target=target)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, len(dataset)),
        shuffle=shuffle,
        drop_last=True
    )
    return dataset, dataloader


steven_model = KanResWide_X().to(conf.DEVICE)
steven_model.load_state_dict(torch.load(conf.TRAINED_MODEL_PATH, map_location=conf.DEVICE))
mse = torch.nn.MSELoss()


def validate_with_steven_model(ecg, rr):
    steven_model.eval()
    with torch.inference_mode():
        predicted_rr = steven_model(ecg)
        loss = mse(predicted_rr, rr)
        print(f'Predicted RR: {predicted_rr}. True RR: {rr}')
    return loss



