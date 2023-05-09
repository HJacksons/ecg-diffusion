from datasets import PTB_Dataset
import configuration as conf
import zipfile
import os

# To prepare the data place the dataset .zip file
# in the data directory and run this script

raw_data_path = os.path.abspath('PTB.zip')
with zipfile.ZipFile(raw_data_path, 'r') as zip_ref:
    zip_ref.extractall()

transformer = PTB_Dataset(conf.DATASET_PATH)
transformer.convert()
