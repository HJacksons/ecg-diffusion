from networks.DiffWave import DiffWave
from contracts.pod import PodContract
import configuration as conf
from tqdm.auto import tqdm
import torch.nn as nn
import numpy as np
import torch


class DiffWavePod(PodContract):
    def __init__(self, lr):
        self.model = DiffWave(
            conf.HYPER_PARAMETERS['residual_layers'],
            conf.HYPER_PARAMETERS['residual_channels'],
            dilation_cycle_length=10,
            n_mels=80,  # just for conditional
            noise_schedule=np.linspace(1e-4, 0.05, conf.HYPER_PARAMETERS['time_steps']).tolist(),
            unconditional=True
        ).to(conf.DEVICE)
        self.loss_fn = nn.L1Loss()
        self.noise_schedule = np.linspace(1e-4, 0.05, conf.HYPER_PARAMETERS['time_steps']).tolist()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.inference_noise_schedule = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5]
        self.autocast = torch.autocast(device_type=conf.DEVICE, enabled=True)
        self.scaler = torch.cuda.amp.GradScaler(enabled=conf.DEVICE == "cuda")

        beta = np.array(self.noise_schedule)
        noise_level = np.cumprod(1 - beta)
        self.noise_level = torch.tensor(noise_level.astype(np.float32))

    def batch_processing(self, batch, leadsI_VIII, feature):
        self.optimizer.zero_grad()

        noise_level = self.noise_level.to(device=conf.DEVICE)

        with self.autocast:
            t = torch.randint(0, len(self.noise_schedule), [leadsI_VIII.shape[0]], device=conf.DEVICE)
            noise_scale = noise_level[t].unsqueeze(1).unsqueeze(1)
            noise_scale_sqrt = noise_scale ** 0.5
            noise = torch.randn_like(leadsI_VIII)
            noisy_audio = noise_scale_sqrt * leadsI_VIII + (1.0 - noise_scale) ** 0.5 * noise

            predicted = self.model(noisy_audio, t)
            loss = self.loss_fn(noise, predicted.squeeze(1))

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.cpu()

    def sampling(self, load_pretrained_model=False):
        trained_model_path = f"{conf.MODELS_FOLDER}/Diffwave_epoch900.pt"

        if load_pretrained_model:
            self.model.load_state_dict(
                torch.load(
                    trained_model_path,
                    map_location=torch.device(conf.DEVICE)
                )
            )

        self.model.eval()
        with torch.no_grad():
            training_noise_schedule = np.array(self.noise_schedule)
            inference_noise_schedule = np.array(self.inference_noise_schedule)

            talpha = 1 - training_noise_schedule
            talpha_cum = np.cumprod(talpha)

            beta = inference_noise_schedule
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            T = []
            for s in range(len(inference_noise_schedule)):
                for t in range(len(training_noise_schedule) - 1):
                    if talpha_cum[t + 1] <= alpha_cum[s] <= talpha_cum[t]:
                        twiddle = (talpha_cum[t] ** 0.5 - alpha_cum[s] ** 0.5) / (
                                talpha_cum[t] ** 0.5 - talpha_cum[t + 1] ** 0.5)
                        T.append(t + twiddle)
                        break
            T = np.array(T, dtype=np.float32)

            signal = torch.randn(8, 5000, device=conf.DEVICE)

            # Sapling loop
            for n in tqdm(range(len(alpha) - 1, -1, -1), desc='Sampling', leave=False, position=0):
                c1 = 1 / alpha[n] ** 0.5
                c2 = beta[n] / (1 - alpha_cum[n]) ** 0.5
                signal = c1 * (signal - c2 * self.model(signal, torch.tensor([T[n]], device=conf.DEVICE)).squeeze(1))

                if n > 0:
                    noise = torch.randn_like(signal)
                    sigma = ((1.0 - alpha_cum[n - 1]) / (1.0 - alpha_cum[n]) * beta[n]) ** 0.5
                    signal += sigma * noise

                signal = torch.clamp(signal, -1.0, 1.0)

        return signal

    def validation(self):
        pass
