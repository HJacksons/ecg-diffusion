from src.networks.UNet import UNet
from src.contracts.pod import PodContract
from diffusers import DDPMScheduler
import src.configuration as conf
import torch.nn as nn
import src.helpers as helpers
import torch


class UNetPod(PodContract):
    def __init__(self, lr):
        self.model = UNet().to(conf.DEVICE)
        self.loss_fn = nn.L1Loss()
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=300, beta_schedule='squaredcos_cap_v2')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999))

    def batch_processing(self, batch, leadsI_VIII, feature):
        self.optimizer.zero_grad()

        noise = torch.randn_like(leadsI_VIII)

        timesteps = torch.randint(0, 299, (leadsI_VIII.shape[0],)).long().to(device=conf.DEVICE)
        noisy_x = self.noise_scheduler.add_noise(leadsI_VIII, noise, timesteps)
        predicted_noise = self.model(noisy_x, timesteps)
        loss = self.loss_fn(predicted_noise, noise)

        loss.backward()
        self.optimizer.step()

        return loss.cpu()

    def sampling(self, epoch):
        x = torch.randn(1, 8, 5000).to(device=conf.DEVICE)

        self.model.eval()
        for i, t in enumerate(self.noise_scheduler.timesteps):
            with torch.no_grad():
                residual = self.model(x, t.to(device=conf.DEVICE))
                # Update sample with step
            x = self.noise_scheduler.step(residual, t, x).prev_sample

        plot_path = f'{conf.PLOTS_FOLDER}/{conf.MODEL}-epoch-{epoch}'
        helpers.create_and_save_plot(x[0].cpu().detach().numpy(), filename=plot_path)

        return plot_path

    def validation(self):
        pass
