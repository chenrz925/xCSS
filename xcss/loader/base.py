from typing import Text
from abc import ABCMeta, abstractmethod
from dependency_injector import containers, providers
from pathlib import Path


class BaseLoader(object, metaclass=ABCMeta):
    def __init__(self, work_dir: Text, *args, **kwargs):
        self.work_dir = Path(work_dir)

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("__len__ is not implemented")

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("__getitem__ is not implemented")

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError("__iter__ is not implemented")

    @property
    @abstractmethod
    def images_dir(self):
        raise NotImplementedError("images_dir is not implemented")
