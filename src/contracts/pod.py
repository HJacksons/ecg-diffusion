from abc import ABC, abstractmethod


class PodContract(ABC):
    @abstractmethod
    def __init__(self, lr):
        pass

    @abstractmethod
    def batch_processing(self, batch, leadsI_VIII, feature):
        pass

    @abstractmethod
    def sampling(self, load_pretrained_model=False):
        pass

    @abstractmethod
    def validation(self):
        pass
