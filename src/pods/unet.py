from src.networks.UNet import UNet_conditional
from diffusers import DDPMScheduler
import src.configuration as conf
import torch.nn as nn
import torch


class UNetPod:
    def __init__(self, lr):
        self.model = UNet_conditional(num_classes=130).to(conf.DEVICE)
        self.loss_fn = nn.MSELoss()
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=300, beta_schedule='squaredcos_cap_v2')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))

    def batch_processing(self, leadsI_VIII, rr):
        self.optimizer.zero_grad()

        rr = rr.squeeze().to(device=conf.DEVICE)
        leadsI_VIII = leadsI_VIII.to(device=conf.DEVICE)
        noise = torch.randn_like(leadsI_VIII)

        timesteps = torch.randint(0, 299, (leadsI_VIII.shape[0],)).long().to(device=conf.DEVICE)
        noisy_x = self.noise_scheduler.add_noise(leadsI_VIII, noise, timesteps)
        predicted_noise = self.model(noisy_x, timesteps, rr)
        loss = self.loss_fn(predicted_noise, noise)

        loss.backward()
        self.optimizer.step()

        return loss.cpu()

    def post_batch_processing(self):
        #  We do not run any other operations other than batch for UNet
        pass
