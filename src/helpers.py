from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from datasets import PTB_Dataset, TensorDataset
from math import ceil
from matplotlib import pyplot as plt
import configuration as conf
from torch import autograd
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


# Adapted from https://github.com/caogang/wgan-gp/blob/master/gan_toy.py
def calc_gradient_penalty(net_dis, real_data, fake_data, batch_size, lmbda, use_cuda=False):
    # Compute interpolation factors
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    # Interpolate between real and fake data.
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    # Evaluate discriminator
    disc_interpolates = net_dis(interpolates)

    # Obtain gradients of the discriminator with respect to the inputs
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else
                              torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    # Compute MSE between 1.0 and the gradient of the norm penalty to make discriminator
    # to be a 1-Lipschitz function.
    gradient_penalty = lmbda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def compute_leadIII_aVR_aVL_aVF(leadI, leadsII):
    lead_III_value = leadsII - leadI # (lead II value) - (lead I value)
    lead_aVR_value = -(leadI + leadsII) / 2 # -0.5*(lead I value + lead II value)
    lead_aVL_value = leadI - leadsII / 2 # lead I value - 0.5 * lead II value
    lead_aVF_value = leadsII - leadI / 2 # lead II value - 0.5 * lead I value

    return lead_III_value, lead_aVR_value, lead_aVL_value, lead_aVF_value



def create_and_save_standardized_12_lead_ecg_plot(number_of_leads_as_input, leads, generated_leads, filename, plot_seconds=10, plot_columns=2, plot_range=1.8, file_extension='.pdf'):
    # code adapted from https://github.com/dy1901/ecg_plot
    def _ax_plot(ax, x, y1, y2, seconds=10, amplitude_ecg=1.8, time_ticks=0.2, plot_second_signal=True):
        ax.set_xticks(np.arange(0, 11, time_ticks))
        ax.set_yticks(np.arange(-ceil(amplitude_ecg), ceil(amplitude_ecg), 1.0))
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_ylim(-amplitude_ecg, amplitude_ecg)
        ax.set_xlim(0, seconds)
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
        ax.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
        ax.plot(x, y1, linewidth=1.5, color="black")
        if plot_second_signal:
            ax.plot(x, y2, linewidth=2, color="blue", linestyle='dotted')

    number_of_leads = 12
    voltage=20
    sample_rate=500
    speed=50
    lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    lead_order = list(range(0, number_of_leads))
    seconds = plot_seconds if plot_seconds is not None else len(leads[0]) / sample_rate
    plt.rc('axes', titleweight='bold')
    fig, ax = plt.subplots(
        ceil(number_of_leads/plot_columns), plot_columns,
        sharex=True, sharey=True,
        figsize=((speed/25.4)*seconds*plot_columns, (4.1*voltage/25.4)*number_of_leads/plot_columns))
    fig.subplots_adjust(hspace=0.01, wspace=0.02, left=0.01, right=0.98, bottom=0.06, top=0.95)

    step = 1.0 / sample_rate
    for i in range(0, number_of_leads):
        if(plot_columns == 1):
            t_ax = ax[i]
        else:
            t_ax = ax[i // plot_columns, i % plot_columns]
        t_lead = lead_order[i]
        t_ax.set_title(lead_index[t_lead], y=1.0, pad=-14)
        t_ax.tick_params(axis='x', rotation=90)
        plot_second_signal=generated_leads is not None and not (lead_index[t_lead] == "I" or (number_of_leads_as_input == 2 and lead_index[t_lead] == "II"))
        _ax_plot(t_ax, np.arange(0, len(leads[t_lead])*step, step), 
                 leads[t_lead], generated_leads[t_lead] if generated_leads is not None else None, 
                 seconds, amplitude_ecg=plot_range, plot_second_signal=plot_second_signal)
    fig.savefig(Path(filename + file_extension))
    plt.close(fig)
