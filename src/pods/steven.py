from src.networks.Steven import KanResWide_X
from src.contracts.pod import PodContract
import src.configuration as conf
import src.helpers as helpers
import torch.optim as optim
import torch.nn as nn
import torch


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

    def batch_processing(self, batch, leadsI_VIII, feature):
        self.optimizer.zero_grad()

        feature = feature[:,feature_index[conf.FEATURE]].float().unsqueeze(1).to(device=conf.DEVICE)
        leadsI_VIII = leadsI_VIII.to(device=conf.DEVICE)
        
        predicted_feature = self.model(leadsI_VIII)
        loss = self.loss_fn(predicted_feature, feature)

        loss.backward()
        self.optimizer.step()

        return loss.cpu()

    def sampling(self, epoch):
        return None

    def validation(self):
        _, validation_dataloader = helpers.get_dataloader(target='validation', batch_size=32, shuffle=True)
        self.model.eval()
        with torch.inference_mode():
            validation_loss_average = 0
            for batch, (leadsI_VIII, feature) in enumerate(validation_dataloader):
                feature = feature[:,feature_index[conf.FEATURE]].float().unsqueeze(1).to(device=conf.DEVICE)
                leadsI_VIII = leadsI_VIII.to(device=conf.DEVICE)
                predicted_feature = self.model(leadsI_VIII)

                loss = self.loss_fn(predicted_feature, feature)

                validation_loss_average += loss.cpu()

            validation_loss_average /= len(validation_dataloader)
            return validation_loss_average
