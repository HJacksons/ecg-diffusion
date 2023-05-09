from src.networks.Steven import KanResWide_X
from src.contracts.pod import PodContract

import src.configuration as conf
import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random


feature_index = {
    'rr': 0,
    'qrs': 1, 
    'pr': 2,
    'qt': 3,
    'VentricularRate': 4, 
    'R_Peak_i': 5, 
    'R_Peak_ii': 6, 
    'R_Peak_v1': 7, 
    'R_Peak_v2': 8, 
    'R_Peak_v3': 9,
    'R_Peak_v4': 10, 
    'R_Peak_v5': 11, 
    'R_Peak_v6': 12
}

class StevenPod(PodContract):
    def __init__(self, lr):
        self.model = KanResWide_X(input=8, output_size=1).to(conf.DEVICE)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.NAdam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.004)

    def batch_processing(self, leadsI_VIII, feature):
        self.optimizer.zero_grad()

        feature = feature[:,feature_index[conf.FEATURE]].float().unsqueeze(1).to(device=conf.DEVICE)
        leadsI_VIII = leadsI_VIII.to(device=conf.DEVICE)
        
        predicted_feature = self.model(leadsI_VIII)
        loss = self.loss_fn(predicted_feature, feature)

        loss.backward()
        self.optimizer.step()

        return loss.cpu()

    def post_batch_processing(self):
        #  We do not run any other operations other than batch for UNet
        pass
