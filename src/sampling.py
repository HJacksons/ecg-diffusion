import torch
from networks.Steven import KanResWide_X
from networks.UNet import UNet
from diffusers import DDPMScheduler
import pandas as pd
import numpy as np
import helpers
import configuration as conf

UNET_PATH = 'pretrained_models/UNet_epoch980.pt'
NUMBER_OF_FILES_TO_GENERATE = 1
STEVEN_MODELS = ['rr', 'qrs', 'qt', 'ventr_rate', 'r_peak_i', 'r_peak_v1']

# Sampling for UNet
noise_scheduler = DDPMScheduler(num_train_timesteps=200, beta_schedule='squaredcos_cap_v2')

# Load checkpoints
model = UNet().to(device=conf.DEVICE)
model.load_state_dict(torch.load(UNET_PATH, map_location=torch.device(conf.DEVICE)))

steven_net = KanResWide_X().to(device=conf.DEVICE)

# Sampling loop
for file_index in range(NUMBER_OF_FILES_TO_GENERATE):
    x = torch.randn(1, 8, 5000).to(device=conf.DEVICE)

    model.eval()
    for i, t in enumerate(noise_scheduler.timesteps):
        with torch.no_grad():
            residual = model(x, t.to(device=conf.DEVICE))
            # Update sample with step
        x = noise_scheduler.step(residual, t, x).prev_sample

    # create dataframe with the generated leads
    x_df = pd.DataFrame(torch.t(x.squeeze()).cpu().numpy(), columns=["I", "II", "v1", "v2", "v3", "v4", "v5", "v6"])

    # predict features
    for feature in STEVEN_MODELS:
        steven_net.load_state_dict(
            torch.load(f'pretrained_models/{feature}.pt', map_location=torch.device(conf.DEVICE)))
        steven_net.eval()
        with torch.inference_mode():
            predicted_feature = steven_net(x)

        # add the feature to the existing dataframe
        predicted_feature = torch.round(predicted_feature).int().cpu().numpy().squeeze().squeeze()
        # fill with NaN to have the same length as the leads
        x_df[f'{feature}'] = pd.Series([predicted_feature, *([np.nan] * (x_df.shape[0] - 1))])

    # save csv file
    x_df.to_csv(f'generated_files/generated_{file_index}.csv', index=False)

    # create plots
    helpers.create_and_save_plot(x[0].cpu().detach().numpy(), filename=f'generated_plots/ecg{file_index}')
