from pods.pod import ModelPod
import configuration as conf
from tqdm.auto import tqdm
import helpers
import wandb
import torch
import yaml


helpers.create_folder_if_not_exists(conf.PLOTS_FOLDER)
helpers.create_folder_if_not_exists(conf.MODELS_FOLDER)
helpers.create_folder_if_not_exists(conf.GEN_DATA_FOLDER)

# Init WANDB if needed
if conf.USE_WEIGHTS_AND_BIASES:
    wandb = helpers.init_wandb()

# Prep for training or tuning
if conf.ACTION in ("train", "tune"):
    helpers.fix_seed()

# Init the pod that houses the models
model_container = ModelPod()


def training_loop():
    # Data loaders
    train_dataset, train_dataloader = helpers.get_dataloader(
        target='train',
        batch_size=model_container.get_batch_size()
    )

    # Train mode
    model_container.pod.model.train()

    # Training loop
    for epoch in tqdm(range(conf.EPOCHS), desc='Epochs', colour='green', leave=False, position=0):
        train_loss_history = []

        batch = 0
        for (leadsI_VIII, feature) in tqdm(train_dataloader, desc='Batch', leave=False, position=1):
            leadsI_VIII = leadsI_VIII.to(device=conf.DEVICE)

            # Run batch operations
            train_loss = model_container.pod.batch_processing(batch, leadsI_VIII, feature)

            if train_loss:
                train_loss_history.append(train_loss)

            batch += 1
        # Record average training loss
        train_loss_average = sum(train_loss_history)/len(train_loss_history)

        # Run sampling operations
        plot_filename = model_container.pod.sampling(epoch=epoch)

        # Run validation
        validation_loss_average = model_container.pod.validation()

        # Report results to WandB
        if conf.USE_WEIGHTS_AND_BIASES:
            if train_loss_average:
                wandb.log({"t-loss": train_loss_average})
            if validation_loss_average:
                wandb.log({"v-loss": validation_loss_average})
            if plot_filename:
                wandb.log({"ECG": wandb.Image(plot_filename + ".png")})
        print(f'Epoch: {epoch}. Average train loss: {train_loss_average:04f}.')

        # save model every 10 epochs
        if epoch % 10 == 0:
            model_filename = f"{conf.MODELS_FOLDER}/{conf.MODEL}_epoch{epoch}.pt"
            torch.save(model_container.pod.model.state_dict(), model_filename)

            if conf.USE_WEIGHTS_AND_BIASES:
                wandb.log_artifact(model_filename, name=f'model_epoch_{epoch}', type='Model')


# Run action
if conf.ACTION == "train":
    training_loop()
elif conf.ACTION == "tune" and conf.USE_WEIGHTS_AND_BIASES:
    with open('../sweep_conf.yaml', 'r') as file:
        sweep_configuration = yaml.safe_load(file)

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=conf.WANDB_PROJECT)
    wandb.agent(sweep_id, function=training_loop, count=10)
