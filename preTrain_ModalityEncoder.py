import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append('models')
sys.path.append('dataset')

from dataset.img_dataset import train_loader, test_loader
from utility import init_img_Model, save_model
import torch.nn as nn
import dataset.config as config
from tqdm import tqdm
import torch

import torch
import matplotlib.pyplot as plt
import numpy as np



def train(model, opt, criterion, loader, epoch):
    loop = tqdm(loader, leave=True)
    for idx, (x, labels) in enumerate(loop):
        
        x, labels = x.to(config.DEVICE), labels.to(config.DEVICE)

        outputs = model(x)
        loss = criterion(outputs, labels)

        opt.zero_grad(); loss.backward(); opt.step()


        loop.set_postfix(
            LOSS=loss.item(),
            epoch=epoch,
        )
    return model, opt

def main():
    trn_loader = train_loader()
    # tst_loader = test_loader()
    #-----------------------------------------------------------------
    model, opt = init_img_Model()
    #-----------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    #-----------------------------------------------------------------

    for epoch in range(10):
        model, opt = train(model, opt, criterion, trn_loader, epoch)        
        save_model(model, opt, config.IMAGE_ENCODER_checkpoints)



if __name__ == "__main__":
    main()