from contracts.pod import PodContract
from diffusers import DDPMScheduler
from networks.UNet import UNet
import configuration as conf
from tqdm.auto import tqdm
import torch.nn as nn
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

    def sampling(self, load_pretrained_model=False):
        trained_model_path = f"{conf.MODELS_FOLDER}/UNet_epoch980.pt"

        if load_pretrained_model:
            self.model.load_state_dict(
                torch.load(
                    trained_model_path,
                    map_location=torch.device(conf.DEVICE)
                )
            )

        x = torch.randn(1, 8, 5000).to(device=conf.DEVICE)

        self.model.eval()
        for t in tqdm(self.noise_scheduler.timesteps, desc='Sampling', leave=False, position=0):
            with torch.no_grad():
                residual = self.model(x, t.to(device=conf.DEVICE))

                # Update sample with step
            x = self.noise_scheduler.step(residual, t, x).prev_sample

        return x

    def validation(self):
        pass
