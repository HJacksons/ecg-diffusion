from datasets import TensorDataset
from diffusion_network import UNet
from learning import Diffusion
import configuration as conf
import logging
import helpers
import wandb
import torch

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

if conf.USE_WEIGHTS_AND_BIASES:
    wandb = helpers.init_wandb(
        conf.WANDB_KEY,
        f'{conf.TRAIN_CONFIGURATION["DATASET_OPTION"]}',
        conf.TRAIN_CONFIGURATION
    )

helpers.create_folder_if_not_exists(conf.PLOTS_FOLDER)

if conf.ACTION == 'train':
    helpers.fix_seed()

mse = torch.nn.MSELoss()
diffusion = Diffusion(device=conf.DEVICE)


def train_diffusion(train_dataset: TensorDataset, train_dataloader, validation_dataset: TensorDataset,
                    validation_dataloader):
    model = UNet().to(conf.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.TRAIN_CONFIGURATION['LEARNING_RATE'], betas=(0.5, 0.999))
    train_loss_plot = []
    model.train()

    for epoch in range(conf.TRAIN_CONFIGURATION['EPOCHS']):
        logging.info(f"Starting epoch {epoch}:")
        train_loss_average = 0
        for batch, (leadI, leadsII_VIII) in enumerate(train_dataloader, 0):
            leadsII_VIII = leadsII_VIII.to(conf.DEVICE)
            t = diffusion.sample_time_steps(leadsII_VIII.shape[0]).to(conf.DEVICE)
            x_t, noise = diffusion.noise_images(leadsII_VIII, t)
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_average += loss.cpu()
        train_loss_average /= len(train_dataloader)

        sampled_ecg = diffusion.sample(model, n=1)
        helpers.create_and_save_plot(sampled_ecg[0].cpu().detach().numpy(), filename=f'{conf.PLOTS_FOLDER}/ecg{epoch}')

        if conf.USE_WEIGHTS_AND_BIASES:
            plot_filename = f"{conf.PLOTS_FOLDER}/ecg{epoch}"
            wandb.log({"MSE": train_loss_average})
            wandb.log({"ECG": wandb.Image(plot_filename + ".png")})

        else:
            train_loss_plot.append(train_loss_average)
            print(f'Finished epoch {epoch}. Average loss for this epoch: {train_loss_average:05f}')


if conf.ACTION == "train":
    train_dataset, train_dataloader = helpers.get_dataloader(target='train')
    validation_dataset, validation_dataloader = helpers.get_dataloader(target='validation', batch_size=1, shuffle=False)
    train_diffusion(train_dataset, train_dataloader, validation_dataset, validation_dataloader)
