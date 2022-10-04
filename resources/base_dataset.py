"""
Define the BaseDataset
"""

from abc import ABC
from typing import Text

from torch.utils.data import DataLoader, Dataset


class BaseDataset(Dataset, ABC):
    """
    The BaseDataset
    """
    def __init__(self, path_label: Text = None, dir_image: Text = None):
        self.path_label = path_label
        self.dir_image = dir_image

    def __len__(self):
        return 0

    @staticmethod
    def data2dataloader(dataset, batch_size: int = 4, shuffle: bool = False):
        """
        Create the dataloader from dataset.
        Args:
            dataset: The dataset.
            batch_size: The length of the dataset.
            shuffle: Is shuffle or not.
        Return:
            The dataloader by using DataLoader from torch.utils.data
        """
        train_dataloader = DataLoader(dataset,
                                      batch_size=batch_size,
                                      shuffle=shuffle)
        return train_dataloader
