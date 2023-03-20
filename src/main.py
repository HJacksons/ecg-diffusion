from diffusion_network import UNet_conditional
from diffusers import DDPMScheduler
from learning import Diffusion
import configuration as conf
from tqdm.auto import tqdm
import logging
import helpers
import wandb
import torch
import yaml

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
noise_scheduler = DDPMScheduler(num_train_timesteps=300, beta_schedule='squaredcos_cap_v2')

# Init WANDB if needed
if conf.USE_WEIGHTS_AND_BIASES:
    wandb = helpers.init_wandb()

# Prep for training or tuning
helpers.create_folder_if_not_exists(conf.PLOTS_FOLDER)
if conf.ACTION in ("train", "tune"):
    helpers.fix_seed()


def training_loop():
    # Model and learning method
    model = UNet_conditional(num_classes=130).to(conf.DEVICE)
    diffusion = Diffusion(device=conf.DEVICE)

    _, test_dataloader = helpers.get_dataloader(target='test', batch_size=1, shuffle=False)
    lI_VIII, label = next(iter(test_dataloader))

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

        for batch, (leadsI_VIII, rr) in enumerate(train_dataloader,0):
            rr = rr.squeeze().to(device=conf.DEVICE)
            
            leadsI_VIII = leadsI_VIII.to(device=conf.DEVICE)
            noise = torch.randn_like(leadsI_VIII)

            #t = diffusion.sample_timesteps(leadsI_VIII.shape[0]).to(device)
            #x_t, noise = diffusion.noise_images(leadsI_VIII, t)

            timesteps = torch.randint(0, 299, (leadsI_VIII.shape[0],)).long().to(device=conf.DEVICE)
            noisy_x = noise_scheduler.add_noise(leadsI_VIII, noise, timesteps)
            predicted_noise = model(noisy_x, timesteps, rr)
            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_average += loss.cpu()
        train_loss_average /= len(train_dataloader)

        # Prepare random x to start from, plus some desired labels y
        x = torch.randn(1, 8, 5000).to(device=conf.DEVICE)
        y = label.squeeze().to(device=conf.DEVICE)

        # Sampling loop
        for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

            # Get model pred
            with torch.no_grad():
                residual = model(x, t.to(device=conf.DEVICE), y)  # Again, note that we pass in our labels y

            # Update sample with step
            x = noise_scheduler.step(residual, t, x).prev_sample

        #sampled_ecg = diffusion.sample(model, n=1)

        helpers.create_and_save_plot(lI_VIII[0].cpu().detach().numpy(), x[0].cpu().detach().numpy(), filename=f'{conf.PLOTS_FOLDER}/ecg{epoch}')

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
