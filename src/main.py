from diffusion_network import UNet
from learning import Diffusion
import configuration as conf
import logging
import helpers
import wandb
import torch
import yaml

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# Init WANDB if needed
if conf.USE_WEIGHTS_AND_BIASES:
    wandb = helpers.init_wandb()

# Prep for training or tuning
helpers.create_folder_if_not_exists(conf.PLOTS_FOLDER)
if conf.ACTION in ("train", "tune"):
    helpers.fix_seed()


def training_loop():
    # Model and learning method
    model = UNet().to(conf.DEVICE)
    diffusion = Diffusion(device=conf.DEVICE)

    # Error function and optimizer
    mse = torch.nn.MSELoss()
    lr = wandb.config.learning_rate if conf.USE_WEIGHTS_AND_BIASES else conf.HYPER_PARAMETERS['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    # Data loaders
    bs = wandb.config.batch_size if conf.USE_WEIGHTS_AND_BIASES else conf.HYPER_PARAMETERS['batch_size']
    train_dataset, train_dataloader = helpers.get_dataloader(target='train', batch_size=bs)

    # validation_dataset, validation_dataloader = helpers.get_dataloader(
    #     target='validation',
    #     batch_size=1,
    #     shuffle=False
    # )

    train_loss_plot = []
    model.train()

    # Training loop
    epochs = wandb.config.epochs if conf.USE_WEIGHTS_AND_BIASES else conf.HYPER_PARAMETERS['epochs']
    for epoch in range(epochs):
        logging.info(f"Starting epoch {epoch}/{epochs}:")
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
            wandb.log({
                "MSE": train_loss_average,
                "ECG": wandb.Image(plot_filename + ".png")
            })
        else:
            train_loss_plot.append(train_loss_average)
            print(f'Finished epoch {epoch}. Average loss for this epoch: {train_loss_average:05f}')


# Run action
if conf.ACTION == "train":
    training_loop()
elif conf.ACTION == "tune" and conf.USE_WEIGHTS_AND_BIASES:
    with open('../sweep_conf.yaml', 'r') as file:
        sweep_configuration = yaml.safe_load(file)

    # todo - fix sweep output dir to be in '../data'
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=conf.WANDB_PROJECT)
    wandb.agent(sweep_id, function=training_loop, count=10)
