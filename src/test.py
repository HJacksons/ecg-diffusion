from networks.Steven import KanResWide_X
from pods.pod import ModelPod
import configuration as conf
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import argparse
import helpers
import torch

helpers.create_folder_if_not_exists(conf.PLOTS_FOLDER)
helpers.create_folder_if_not_exists(conf.GEN_DATA_FOLDER)

# Add support for arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--files', type=int, help='Number of files to generate and plot')
parser.add_argument('-n', '--network', type=str, help='Which network to test (unet, diffwave, pulse2pulse)')
args = parser.parse_args()

NUMBER_OF_FILES_TO_GENERATE = 1

if args.files:
    NUMBER_OF_FILES_TO_GENERATE = args.files

if args.network:
    conf.MODEL = args.network

# Init the pod that houses the models
model_container = ModelPod()

# Init the STeven model
steven_net = KanResWide_X().to(device=conf.DEVICE)
STEVEN_MODELS = ['rr', 'qrs', 'qt', 'ventr_rate', 'r_peak_i', 'r_peak_v1']

# Sampling loop
for file_index in tqdm(range(NUMBER_OF_FILES_TO_GENERATE), desc='Files', colour='blue', leave=False, position=1):
    x = model_container.pod.sampling(load_pretrained_model=True)

    if x is None:
        exit(1)

    # create dataframe with the generated leads
    x_df = pd.DataFrame(
        torch.t(x.squeeze()).cpu().numpy(),
        columns=["I", "II", "v1", "v2", "v3", "v4", "v5", "v6"]
    )

    # predict features
    for feature in tqdm(STEVEN_MODELS, desc='Features', colour='green', leave=False, position=0):
        # load Steven model of the feature
        steven_net.load_state_dict(
            torch.load(
                f'{conf.GEN_MODELS_FOLDER}/Steven_{feature}.pt',
                map_location=torch.device(conf.DEVICE)
            )
        )

        # predict feature
        steven_net.eval()
        with torch.inference_mode():
            predicted_feature = steven_net(x)

        # add the feature to the existing dataframe
        predicted_feature = torch.round(predicted_feature).int().cpu().numpy().squeeze().squeeze()

        # fill with NaN to have the same length as the leads
        x_df[f'{feature}'] = pd.Series([predicted_feature, *([np.nan] * (x_df.shape[0] - 1))])

    # save csv file
    x_df.to_csv(f'{conf.GEN_DATA_FOLDER}/{conf.MODEL}-{file_index}.csv', index=False)

    # create plots
    helpers.create_and_save_plot(
        x[0].cpu().detach().numpy(),
        filename=f'{conf.PLOTS_FOLDER}/{conf.MODEL}-ecg-{file_index}'
    )

# realistic ecg plots
NUMBER_OF_PLOTS_TO_GENERATE = NUMBER_OF_FILES_TO_GENERATE

for file_index in tqdm(range(0, NUMBER_OF_PLOTS_TO_GENERATE), desc='Plots', colour='yellow', leave=False, position=2):
    ecg_df = pd.read_csv(f"{conf.GEN_DATA_FOLDER}/{conf.MODEL}-{file_index}.csv")

    lead_I = ecg_df.loc[:, 'I']
    lead_II = ecg_df.loc[:, 'II']
    lead_v1 = ecg_df.loc[:, 'v1']
    lead_v2 = ecg_df.loc[:, 'v2']
    lead_v3 = ecg_df.loc[:, 'v3']
    lead_v4 = ecg_df.loc[:, 'v4']
    lead_v5 = ecg_df.loc[:, 'v5']
    lead_v6 = ecg_df.loc[:, 'v6']
    lead_III, lead_aVR, lead_aVL, lead_aVF = helpers.compute_leadIII_aVR_aVL_aVF(lead_I, lead_II)

    data = np.array([lead_I, lead_II, lead_III, lead_aVR, lead_aVL, lead_aVF,
                     lead_v1, lead_v2, lead_v3, lead_v4, lead_v5, lead_v6])

    helpers.create_and_save_standardized_12_lead_ecg_plot(
        12,
        data * 8,
        generated_leads=None,
        filename=f'{conf.PLOTS_FOLDER}/{conf.MODEL}_{file_index}',
        plot_seconds=10,
        plot_columns=2,
        plot_range=1.8,
        file_extension='.pdf'
    )
