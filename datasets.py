# -*- coding: utf-8 -*-

import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import warnings
import os
from pathlib import Path

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomCrop(224),
    transforms.ToTensor()
])

class D3net_Dataset(Dataset):
    def __init__(self, dir_path, transforms_=None):

        self.dir_path = Path(dir_path)
        self.transform = transforms_
        self.files = [file for file in os.listdir(dir_path)]
            
    def __getitem__(self, index):
        try:
            image = Image.open(self.dir_path / self.files[index])   
            image = to_rgb(image)
            item = self.transform(image)
            return item

        except:
            return self.__getitem__(index + 1)
    
    def __len__(self):
        return len(self.files)


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, path, transform=None, shuffle=True,
                 num_workers=8, batch_size=4, *args, **kwargs):

        if transform is None:
            transform = DEFAULT_TRANSFORM

        super().__init__(
            D3net_Dataset(path, transform),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs
        )

