from abc import ABC, abstractmethod


class PodContract(ABC):
    @abstractmethod
    def __init__(self, lr):
        pass

    @abstractmethod
    def batch_processing(self, leadsI_VIII, rr):
        pass

    @abstractmethod
    def post_batch_processing(self):
        pass
