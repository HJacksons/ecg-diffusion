from networks.Steven import KanResWide_X
from pods.pod import ModelPod
import configuration as conf
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import helpers
import torch

# Init the pod that houses the models
model_container = ModelPod()

NUMBER_OF_FILES_TO_GENERATE = 1
STEVEN_MODELS = ['rr', 'qrs', 'qt', 'ventr_rate', 'r_peak_i', 'r_peak_v1']

steven_net = KanResWide_X().to(device=conf.DEVICE)

# Sampling loop
for file_index in range(NUMBER_OF_FILES_TO_GENERATE):
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
