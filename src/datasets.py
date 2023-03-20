from typing import Literal, Tuple, Union
from torch.utils.data import Dataset
import pandas as pd
import random
import torch
import glob
import os


# https://colab.research.google.com/drive/1KN1BTl1Thhhy87YjV04fYP5L_VlUAZ2F?usp=sharing

class TensorDataset(Dataset):
    DATAFILE_EXTENSION = ".pt"

    def __init__(self, data_dirs: Union[str, list[str]], target: Literal["train", "test", "validation"] = "train"):
        print(f"Loading dataset from {data_dirs}")
        self.target = target
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.data_dir = data_dirs

        # the training files are taken from all the data directories
        self.train_files = []
        for data_dir in data_dirs:
            self.train_files.extend(glob.glob(f"{data_dir}/train/*{TensorDataset.DATAFILE_EXTENSION}"))

        # validation and testing is taken only from the last data directory
        data_dir = data_dirs[-1]
        self.validation_files = glob.glob(f"{data_dir}/validation/*{TensorDataset.DATAFILE_EXTENSION}")
        self.test_files = glob.glob(f"{data_dir}/test/*{TensorDataset.DATAFILE_EXTENSION}")
        if self.train_files is None or self.validation_files is None or self.test_files is None:
            return
        self.train_files.sort()
        self.validation_files.sort()
        self.test_files.sort()

    def __len__(self):
        if self.target == "train":
            return len(self.train_files)
        if self.target == "test":
            return len(self.test_files)
        if self.target == "validation":
            return len(self.validation_files)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.target == "train":
            return torch.load(self.train_files[idx])
        if self.target == "test":
            return torch.load(self.test_files[idx])
        if self.target == "validation":
            return torch.load(self.validation_files[idx])

    def get_sample(self, source: Literal["train", "test", "validation"] = "validation", random_sample=False) -> Tuple[
        torch.Tensor, torch.Tensor
    ]:
        if source == "train":
            idx = 0 if not random_sample else random.randint(0, len(self.train_files) - 1)
            return torch.load(self.train_files[idx])
        if source == "test":
            idx = 0 if not random_sample else random.randint(0, len(self.test_files) - 1)
            return torch.load(self.test_files[idx])
        if source == "validation":
            idx = 0 if not random_sample else random.randint(0, len(self.validation_files) - 1)
            return torch.load(self.validation_files[idx])
        raise ValueError()

    def get_start_index(self, target: Literal["train", "test", "validation"] = "test") -> int:
        if target == "train":
            return 0
        if target == "validation":
            return len(self.train_files)
        if target == "test":
            return len(self.train_files) + len(self.validation_files)
        raise ValueError()

    def convert_to_millivolts(self, input):
        return input


class Synthetic_Dataset(TensorDataset):
    MAX_VALUE = 5011
    header = [x for x in range(8)]

    def __init__(self, data_dirs: Union[str, list[str]], target: Literal["train", "test", "validation"] = "train"):
        super().__init__(data_dirs, target)

    def convert_to_millivolts(self, input):
        return input / 1000

    @staticmethod
    def convert_input(input):
        return input / Synthetic_Dataset.MAX_VALUE

    @staticmethod
    def convert_output(tensor_out):
        return tensor_out * Synthetic_Dataset.MAX_VALUE

    def max_value(self):
        max_value = 0
        for folder in ['train', 'test', 'validation']:
            for data_dir in self.data_dir:
                for filename in glob.glob(f'{data_dir}/{folder}/*.csv'):
                    print(filename)
                    temp_df = pd.read_csv(filename, sep=" ", names=Synthetic_Dataset.header)
                    temp_tensor_in = torch.tensor(temp_df.iloc[:, 0], dtype=torch.float32).unsqueeze(0)
                    temp_tensor_out = torch.tensor(temp_df.iloc[:, 1:9].values, dtype=torch.float32).t()
                    max_in = torch.max(temp_tensor_in).item()
                    min_in = torch.min(temp_tensor_in).item()
                    max_out = torch.max(temp_tensor_out).item()
                    min_out = torch.min(temp_tensor_out).item()
                    max_value = max(max_value, max_in, max_out, abs(min_in), abs(min_out))
        print(max_value)

    def convert(self):
        for folder in ['train', 'test', 'validation']:
            for data_dir in self.data_dir:
                for filename in glob.glob(f'{data_dir}/{folder}/*.asc'):
                    print(filename)
                    ecg_index = int(os.path.basename(filename).removesuffix('.asc'))
                    temp_df = pd.read_csv(filename, sep=" ", names=Synthetic_Dataset.header)
                    temp_tensor_in = torch.tensor(Synthetic_Dataset.convert_input(temp_df.iloc[:, 0]),
                                                  dtype=torch.float32).unsqueeze(0)
                    temp_tensor_out = torch.tensor(Synthetic_Dataset.convert_input(temp_df.iloc[:, 1:9].values),
                                                   dtype=torch.float32).t()
                    temp_tensor_pair = (temp_tensor_in, temp_tensor_out)
                    torch.save(temp_tensor_pair, f'{data_dir}/{folder}/{str(ecg_index).zfill(5)}.pt')


class PTB_Dataset(TensorDataset):
    MAX_VALUE = 33  # 32.715999603271484
    header = ["I", "II", "III", "aVF", "aVR", "aVL", "V1", "V2", "V3", "V4", "V5", "V6", "egc_id", "rr_interval"]

    def __init__(self, data_dirs: Union[str, list[str]], target: Literal["train", "test", "validation"] = 'train'):
        super().__init__(data_dirs, target)

    @staticmethod
    def convert_input(input):
        return input / PTB_Dataset.MAX_VALUE

    @staticmethod
    def convert_output(tensor_out):
        return tensor_out * PTB_Dataset.MAX_VALUE

    def max_value(self):
        max_value = 0
        for folder in ['train', 'test', 'validation']:
            for data_dir in self.data_dir:
                for filename in glob.glob(f'{data_dir}/{folder}/*.csv'):
                    print(filename)
                    temp_df = pd.read_csv(filename, index_col=0, header=0, names=PTB_Dataset.header)
                    temp_tensor_in = torch.tensor(temp_df.iloc[:, 0], dtype=torch.float32).unsqueeze(0)
                    temp_tensor_out = torch.tensor(temp_df.iloc[:, [1, 6, 7, 8, 9, 10, 11]].values, dtype=torch.float32)
                    max_in = torch.max(temp_tensor_in).item()
                    min_in = torch.min(temp_tensor_in).item()
                    max_out = torch.max(temp_tensor_out).item()
                    min_out = torch.min(temp_tensor_out).item()
                    max_value = max(max_value, max_in, max_out, abs(min_in), abs(min_out))
        print(max_value)

    def convert(self):
        for folder in ['train', 'test', 'validation']:
            for data_dir in self.data_dir:
                for filename in glob.glob(f'{data_dir}/{folder}/*.csv'):
                    ecg_index = int(os.path.basename(filename).removeprefix('ecg').removesuffix('.csv'))
                    temp_df = pd.read_csv(filename)
                    # temp_tensor_in = torch.tensor(PTB_Dataset.convert_input(temp_df.iloc[:, 0]), dtype=torch.float32).unsqueeze(0)
                    temp_tensor_in = torch.tensor(
                        PTB_Dataset.convert_input(temp_df.iloc[:, [0, 1, 6, 7, 8, 9, 10, 11]].values),
                        dtype=torch.float32).t()
                    temp_tensor_out = torch.tensor(temp_df.iloc[:, [13]].values)
                    temp_tensor_pair = (temp_tensor_in, temp_tensor_out[0].long())
                    torch.save(temp_tensor_pair, f'{data_dir}/{folder}/{str(ecg_index).zfill(5)}.pt')


# import zipfile
# with zipfile.ZipFile('PTB.zip', 'r') as zip_ref:
#     zip_ref.extractall()


# dataset_folder = 'synthetic'
# transformer = Synthetic_Dataset(dataset_folder)
# transformer.convert()

# dataset_folder = 'PTB'
# transformer = PTB_Dataset(dataset_folder)
# transformer.convert()

# dataset_folder = 'PTB_pathologic'
# transformer = PTB_Dataset(dataset_folder)
# transformer.convert()
# transformer.max_value()
