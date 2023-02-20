from network import Pulse2pulseGenerator
from datasets import PTB_Dataset, TensorDataset

import torch
import numpy as np
from pathlib import Path
from typing import Tuple
import random
from datetime import datetime

ACTION = 'train'  # train | test | generate_outputs

# CONFIGURATION
train_configuration = {
    'DATASET_OPTION': ['../PTB'],  # synthetic | PTB | PTB_pathologic | ['synthetic', 'PTB']
    'NETWORK_OPTION': 'AE',  # GAN | AE
    'TRAINING_MODE': 'simple',  # simple | transfer
    'LEARNING_RATE': 0.0001,
    'MODEL_SIZE': 16,
    'BATCH_SIZE': 32,
    'EPOCHS': 200,
    'DISCRIMINATOR_PATCH_SIZE': 1000
}
# CONFIGURATION END


def fix_seed():
    # torch.use_deterministic_algorithms(True)
    # import os
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    random_seed = 123
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if ACTION == 'train':
    fix_seed()


def get_dataloader(target='train', batch_size=train_configuration['BATCH_SIZE'], shuffle=True) -> Tuple[
    TensorDataset, torch.utils.data.DataLoader]:
    dataset = PTB_Dataset(data_dirs=train_configuration['DATASET_OPTION'], target=target)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle,
                                             drop_last=True)
    return dataset, dataloader


def train_ae(train_dataset: TensorDataset, train_dataloader, validation_dataset: TensorDataset, validation_dataloader):
    generator = Pulse2pulseGenerator(model_size=train_configuration['MODEL_SIZE']).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(generator.parameters(), lr=train_configuration['LEARNING_RATE'], betas=(0.5, 0.999))
    train_loss_plot = []

    for epoch in range(train_configuration['EPOCHS']):
        generator.train()
        train_loss_average = 0
        for batch, (leadI, leadsII_VIII) in enumerate(train_dataloader, 0):
            if batch % 20 == 0: print(f'Batch: {batch}, time {datetime.now()}')

            leadI = leadI.to(device)
            leadsII_VIII = leadsII_VIII.to(device)

            output = generator(leadI)
            train_criterion = criterion(train_dataset.convert_to_millivolts(train_dataset.convert_output(output)),
                                        train_dataset.convert_to_millivolts(train_dataset.convert_output(leadsII_VIII)))
            train_loss_average += train_criterion.data.cpu()
            optimizer.zero_grad()
            train_criterion.backward()
            optimizer.step()

        train_loss_average /= len(train_dataloader)
        train_loss_plot.append(train_loss_average)


if ACTION == "train":
    train_dataset, train_dataloader = get_dataloader(target='train')
    validation_dataset, validation_dataloader = get_dataloader(target='validation', batch_size=1, shuffle=False)
    train_ae(train_dataset, train_dataloader, validation_dataset, validation_dataloader)
