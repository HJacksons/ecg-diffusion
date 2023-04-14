from src.networks.DiffWave import DiffWave
import torch.nn as nn
import configuration as conf
from tqdm.auto import tqdm
import helpers
import wandb
import torch
import yaml
import numpy as np 


# Init WANDB if needed
if conf.USE_WEIGHTS_AND_BIASES:
    wandb = helpers.init_wandb()

# Prep for training or tuning
helpers.create_folder_if_not_exists(conf.PLOTS_FOLDER)
helpers.create_folder_if_not_exists(conf.MODELS_FOLDER)
if conf.ACTION in ("train", "tune"):
    helpers.fix_seed()


def training_loop():

    noise_schedule = np.linspace(1e-4, 0.05, 50).tolist()
    inference_noise_schedule = [0.0001, 0.001, 0.01, 0.05, 0.2, 0.5]

    # Model and learning method
    model = DiffWave(conf.HYPER_PARAMETERS['residual_layers'], 
                     conf.HYPER_PARAMETERS['residual_channels'], 
                     dilation_cycle_length=10, 
                     n_mels=80,  # just for conditional
                     noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(), 
                     unconditional=True).to(conf.DEVICE)


    # Error function and optimizer
    loss_fn = nn.L1Loss()
    lr = wandb.config.learning_rate if conf.USE_WEIGHTS_AND_BIASES else conf.HYPER_PARAMETERS['learning_rate']
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    autocast = torch.cuda.amp.autocast(enabled=True)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    beta = np.array(noise_schedule)
    noise_level = np.cumprod(1 - beta)
    noise_level = torch.tensor(noise_level.astype(np.float32))

    # Data loaders
    bs = wandb.config.batch_size if conf.USE_WEIGHTS_AND_BIASES else conf.HYPER_PARAMETERS['batch_size']
    train_dataset, train_dataloader = helpers.get_dataloader(target='train', batch_size=bs)

    model.train()

    # Training loop
    for epoch in tqdm(range(conf.EPOCHS), desc='Epochs', colour='green', leave=False, position=0):
        train_loss_average = 0

        for (leadsI_VIII, rr) in tqdm(train_dataloader, desc='Batch', leave=False, position=1):
            optimizer.zero_grad()
            
            leadsI_VIII = leadsI_VIII.to(device=conf.DEVICE)
            noise_level = noise_level.to(device=conf.DEVICE)

            with autocast:
                t = torch.randint(0, len(noise_schedule), [leadsI_VIII.shape[0]], device=conf.DEVICE)
                noise_scale = noise_level[t].unsqueeze(1).unsqueeze(1)
                noise_scale_sqrt = noise_scale**0.5
                noise = torch.randn_like(leadsI_VIII)
                noisy_audio = noise_scale_sqrt * leadsI_VIII + (1.0 - noise_scale)**0.5 * noise

                predicted = model(noisy_audio, t)
                loss = loss_fn(noise, predicted.squeeze(1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1e9)
            scaler.step(optimizer)
            scaler.update()

            train_loss_average += loss.cpu()
        train_loss_average /= len(train_dataloader)

        # Sampling
        model.eval()
        with torch.no_grad():
            training_noise_schedule = np.array(noise_schedule)
            inference_noise_schedule = np.array(inference_noise_schedule) 

            talpha = 1 - training_noise_schedule
            talpha_cum = np.cumprod(talpha)

            beta = inference_noise_schedule
            alpha = 1 - beta
            alpha_cum = np.cumprod(alpha)

            T = []
            for s in range(len(inference_noise_schedule)):
                for t in range(len(training_noise_schedule) - 1):
                    if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
                        twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
                        T.append(t + twiddle)
                        break
            T = np.array(T, dtype=np.float32)

            audio = torch.randn(8, 5000, device=conf.DEVICE)
            noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(conf.DEVICE)

            for n in range(len(alpha) - 1, -1, -1):
                c1 = 1 / alpha[n]**0.5
                c2 = beta[n] / (1 - alpha_cum[n])**0.5
                audio = c1 * (audio - c2 * model(audio, torch.tensor([T[n]], device=conf.DEVICE)).squeeze(1))
                if n > 0:
                    noise = torch.randn_like(audio)
                    sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
                    audio += sigma * noise
                audio = torch.clamp(audio, -1.0, 1.0)

        helpers.create_and_save_plot(audio[0].cpu().detach().numpy(), filename=f'{conf.PLOTS_FOLDER}/ecg{epoch}')

        if conf.USE_WEIGHTS_AND_BIASES:
            plot_filename = f"{conf.PLOTS_FOLDER}/ecg{epoch}"
            wandb.log({
                "MSE": train_loss_average,
                "ECG": wandb.Image(plot_filename + ".png")
            })
        print(f'Epoch: {epoch}. Average train loss: {train_loss_average:04f}.') 
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
