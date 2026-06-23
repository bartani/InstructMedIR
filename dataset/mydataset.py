import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import config
import os
from torchvision.utils import save_image
import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
from degradation import task_degradation


class train_dataset(Dataset):
    def __init__(self, imgs, aug=True):
        self.files = imgs
        self.lendata = len(self.files)
        self.aug = aug
        self.points = config.get_Points()

    def __len__(self):
        return self.lendata*1

    def __getitem__(self, index):
        _path = self.files[index % self.lendata]
        modality_path = Path(_path)
        modality_name = modality_path.parent.name

        # generate mask
        mask = config.get_mask_based_on_modality_name(modality_name).float()

        # excite degradation
        selected_task = random.choice(config.tasks)        
        delta = self.points[config.tasks.index(selected_task)]

        # generate clean and degraded modality images
        clean = Image.open(_path).convert('RGB')
        degraded = task_degradation(clean, config.tasks.index(selected_task))

        if self.aug:
            clean, degraded = config.apply_transforms(clean, degraded)
       
        clean = config.tfms(clean)
        degraded = config.tfms(degraded)

        
        
        return clean, degraded, delta, mask


def train_loader():
    myds = train_dataset(config.get_image_files(config.TRAIN_PTH+"/*"))
    loader = DataLoader(
        myds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    return loader

def test_loader():
    myds = train_dataset(config.get_image_files(config.TEST_PATH+"/*"), aug=False)
    loader = DataLoader(
        myds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    return loader


if __name__ == "__main__":
    loader = train_loader()
    
    x, y, delta, mask = next(iter(loader))
    print(x.shape)
    print(y.shape)

    print(delta.shape)

    print(mask.shape)
    print(mask)

    save_image(y*.5+.5, "outcomes/datasample/sample1.png")




