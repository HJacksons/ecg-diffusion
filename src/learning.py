import torch
import logging
from tqdm.auto import tqdm
import torch.nn.functional as F

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=0.0001, beta_end=0.02, img_size=5000, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.betas = self.prepare_noise_schedule().to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[t])[:, None, None]
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None]
        Ɛ = torch.randn_like(x) 
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 8, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alphas[t][:, None, None]
                alpha_cumprod = self.alphas_cumprod[t][:, None, None]
                beta = self.betas[t][:, None, None]
                posterior_variance = self.posterior_variance.gather(-1, t)
                if i > 1:
                    noise = torch.randn_like(x) 
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - beta * predicted_noise / torch.sqrt(1 - alpha_cumprod)) + posterior_variance * noise
        model.train()
        # x = (x.clamp(-1, 1) + 1) / 2
        # x = (x * 33).type(torch.uint8)
        return x