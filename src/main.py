from src.networks.UNet import UNet_conditional
from diffusers import DDPMScheduler
import configuration as conf
from tqdm.auto import tqdm
import logging
import helpers
import wandb
import torch
import yaml

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
noise_scheduler = DDPMScheduler(num_train_timesteps=300, beta_schedule='squaredcos_cap_v2')
VALIDATION_NOISE = torch.randn(1, 8, 5000).to(device=conf.DEVICE)

# Init WANDB if needed
if conf.USE_WEIGHTS_AND_BIASES:
    wandb = helpers.init_wandb()

# Prep for training or tuning
helpers.create_folder_if_not_exists(conf.PLOTS_FOLDER)
helpers.create_folder_if_not_exists(conf.MODELS_FOLDER)
if conf.ACTION in ("train", "tune"):
    helpers.fix_seed()


def training_loop():
    # Model and learning method
    model = UNet_conditional(num_classes=130).to(conf.DEVICE)

    # Error function and optimizer
    mse = torch.nn.MSELoss()
    lr = wandb.config.learning_rate if conf.USE_WEIGHTS_AND_BIASES else conf.HYPER_PARAMETERS['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    # Data loaders
    bs = wandb.config.batch_size if conf.USE_WEIGHTS_AND_BIASES else conf.HYPER_PARAMETERS['batch_size']
    train_dataset, train_dataloader = helpers.get_dataloader(target='train', batch_size=bs)

    model.train()

    # Training loop
    epochs = wandb.config.epochs if conf.USE_WEIGHTS_AND_BIASES else conf.HYPER_PARAMETERS['epochs']
    for epoch in tqdm(range(epochs), desc='Epochs', colour='green', leave=False, position=0):
        train_loss_average = 0

        for (leadsI_VIII, rr) in tqdm(train_dataloader, desc='Batch', leave=False, position=1):
            rr = rr.squeeze().to(device=conf.DEVICE)
            leadsI_VIII = leadsI_VIII.to(device=conf.DEVICE)
            noise = torch.randn_like(leadsI_VIII)

            timesteps = torch.randint(0, 299, (leadsI_VIII.shape[0],)).long().to(device=conf.DEVICE)
            noisy_x = noise_scheduler.add_noise(leadsI_VIII, noise, timesteps)
            predicted_noise = model(noisy_x, timesteps, rr)
            loss = mse(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_average += loss.cpu()
        train_loss_average /= len(train_dataloader)

        # Validation with steven model
        validation_loss_average = 0
        # Generate 100 ECGs from noise
        for validation_index in range(conf.VALIDATION_SAMPLES):
            # always use the same noise (not exactly sure if that's right)
            x = VALIDATION_NOISE
            # choose a random RR value
            y = torch.randint(400, 1500, size=(1,)).to(device=conf.DEVICE)

            # Sampling loop
            for t in tqdm(noise_scheduler.timesteps, desc='Sampling', leave=False, position=2):
                with torch.no_grad():
                    residual = model(x, t.to(device=conf.DEVICE), y)
                # Update sample with step
                x = noise_scheduler.step(residual, t, x).prev_sample
            # sampled_ecg = diffusion.sample(model, n=1)
            validation_loss_average += helpers.validate_with_steven_model(x, y)
        validation_loss_average /= conf.VALIDATION_SAMPLES
        # helpers.create_and_save_plot(x[0].cpu().detach().numpy(), filename=f'{PLOTS_FOLDER}/ecg{epoch}')

        if conf.USE_WEIGHTS_AND_BIASES:
            plot_filename = f"{conf.PLOTS_FOLDER}/ecg{epoch}"
            wandb.log({
                "MSE": train_loss_average,
                "ECG": wandb.Image(plot_filename + ".png"),
                "V-MSE": validation_loss_average
            })
        print(f'Epoch: {epoch}. Average train loss: {train_loss_average:04f}. Average validation loss: {validation_loss_average:04f}')
        print("--------------------------------------------------------------------------------------------------------")

        # save model every 10 epochs
        if epoch % 10 == 0:
            model_filename = f"{conf.MODELS_FOLDER}/model_epoch{epoch}.pt"
            torch.save(model.state_dict(), model_filename)


# Run action
if conf.ACTION == "train":
    training_loop()
elif conf.ACTION == "tune" and conf.USE_WEIGHTS_AND_BIASES:
    with open('../sweep_conf.yaml', 'r') as file:
        sweep_configuration = yaml.safe_load(file)

    # todo - fix sweep output dir to be in '../data'
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=conf.WANDB_PROJECT)
    wandb.agent(sweep_id, function=training_loop, count=10)
