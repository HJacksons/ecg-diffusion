from src.networks.UNet import UNet
from diffusers import DDPMScheduler
import configuration as conf
from tqdm.auto import tqdm
import helpers
import wandb
import torch
import yaml

# Init WANDB if needed
if conf.USE_WEIGHTS_AND_BIASES:
    wandb = helpers.init_wandb()

# Prep for training or tuning
helpers.create_folder_if_not_exists(conf.PLOTS_FOLDER)
helpers.create_folder_if_not_exists(conf.MODELS_FOLDER)
if conf.ACTION in ("train", "tune"):
    helpers.fix_seed()


def training_loop():
    noise_scheduler = DDPMScheduler(num_train_timesteps=200, beta_schedule='squaredcos_cap_v2')

    # Model and learning method
    model = UNet().to(conf.DEVICE)

    # Error function and optimizer
    mae = torch.nn.L1Loss()
    lr = wandb.config.learning_rate if conf.USE_WEIGHTS_AND_BIASES else conf.HYPER_PARAMETERS['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

    # Data loaders
    batch_size = wandb.config.batch_size if conf.USE_WEIGHTS_AND_BIASES else conf.HYPER_PARAMETERS['batch_size']
    train_dataset, train_dataloader = helpers.get_dataloader(target='train', batch_size=batch_size)

    model.train()

    # Training loop
    epochs = wandb.config.epochs if conf.USE_WEIGHTS_AND_BIASES else conf.HYPER_PARAMETERS['epochs']
    for epoch in tqdm(range(epochs), desc='Epochs', colour='green', leave=False, position=0):
        train_loss_average = 0

        for (leadsI_VIII, _) in tqdm(train_dataloader, desc='Batch', leave=False, position=1):
            leadsI_VIII = leadsI_VIII.to(device=conf.DEVICE)
            noise = torch.randn_like(leadsI_VIII)

            timesteps = torch.randint(0, 199, (leadsI_VIII.shape[0],)).long().to(device=conf.DEVICE)
            noisy_x = noise_scheduler.add_noise(leadsI_VIII, noise, timesteps)
            predicted_noise = model(noisy_x, timesteps)
            loss = mae(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_average += loss.cpu()
        train_loss_average /= len(train_dataloader)

        # Sampling loop
        x = torch.randn(1, 8, 5000).to(device=conf.DEVICE)
        model.eval()
        for i, t in enumerate(noise_scheduler.timesteps):
            with torch.no_grad():
                residual = model(x, t.to(device=conf.DEVICE)) 
            # Update sample with step
            x = noise_scheduler.step(residual, t, x).prev_sample
        helpers.create_and_save_plot(x[0].cpu().detach().numpy(), filename=f'{conf.PLOTS_FOLDER}/ecg{epoch}')


        if conf.USE_WEIGHTS_AND_BIASES:
            plot_filename = f"{conf.PLOTS_FOLDER}/ecg{epoch}"
            wandb.log({
                "MAE": train_loss_average,
                "ECG": wandb.Image(plot_filename + ".png")
            })
        print(f'Epoch: {epoch}. Average train loss: {train_loss_average:04f}.')

        # save model every 10 epochs
        if conf.USE_WEIGHTS_AND_BIASES:
            if epoch % 10 == 0:
                model_filename=f"{conf.MODELS_FOLDER}/model"
                torch.save(model.state_dict(), model_filename)
                wandb.log_artifact(model_filename, name=f'model_epoch_{epoch}', type='Model') 


# Run action
if conf.ACTION == "train":
    training_loop()
elif conf.ACTION == "tune" and conf.USE_WEIGHTS_AND_BIASES:
    with open('../sweep_conf.yaml', 'r') as file:
        sweep_configuration = yaml.safe_load(file)

    # todo - fix sweep output dir to be in '../data'
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=conf.WANDB_PROJECT)
    wandb.agent(sweep_id, function=training_loop, count=10)
