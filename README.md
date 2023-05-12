# Electrocardiogram Generation using Diffusion
A latent diffusion model for generating ECG signals

## Getting started

1. Clone the repository
2. Install the dependencies with `pip install -r requirements.txt`
3. Copy the `.env.example` file in the project root and rename it to `.env`
4. Add your environment variables to the file to be able to use some of the features
5. Using the code
   1. Run the `train.py` file for training models
   2. Run the `sample.py` file to generate synthetic data

## Dataset
1. Download the `PTB.zip` file on [Wetransfer](https://wetransfer.com/downloads/38a87f3ef5ff00aeb7e17dc6283ee85120230508163503/a18acc)
2. Place the unzipped file inside the root directory of the project
3. Run the `data_prepare.py` script
   1. Based on the configuration you may need to modify the directory paths found in the `cofigurations.py` script
   2. After the script is done, the data should be split in train, test, and validation directories and each `.csv` file should have a corresponding `.pt` file right below it
4. The extracted and normalized dataset should be in the project root directory

## Trained Models
1. Install [DVC](https://dvc.org/doc/start) on your machine if it is not already installed
2. Run the `dvc pull` command on the project root to get all the latest model files
3. You may need to authenticate with your Google account in order to have access
4. Now you can run the `sample.py` script to generate data from the pre-trained models

## Configurations
All the configuration variables live in the `configurations.py` file. There you can switch between models to train and sample. If you are going to do your own training the hyperparameters of the networks can be found there as well.
1. Training
   1. Choose a model to train in the `configurations.py` script (the options are commented out)
   2. Set up [WandB](https://wandb.ai) credentials if you want artifacts and media to be reported there
   2. Run the `train.py` script, models will be saved each 10th epoch
2. Sampling
   1. Choose a model to sample from in the `configurations.py` script (the options are commented out)
   2. Run the `sample.py` script, generated files and plots will be saved in the `/data` directory

## Collaborators
Oriana Presacan - s372073@oslomet.no\
Frencis Balla - s371513@oslomet.no\
Jackson Herbert Sinamenye - s371140@oslomet.no
