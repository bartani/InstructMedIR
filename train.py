import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append('models')
sys.path.append('dataset')


from dataset.mydataset import train_loader, test_loader
from utility import init_Generator, save_model
from loss import L1, SSIMLoss
import dataset.config as config
from tqdm import tqdm
import torch
from torchvision.utils import save_image

def train(model, opt, loader, criterion, ssim, epoch):
    loop = tqdm(loader, leave=True)
    for idx, (clean, degraded, delta, mask) in enumerate(loop):
        clean, degraded = clean.to(config.DEVICE), degraded.to(config.DEVICE)
        delta, mask = delta.to(config.DEVICE), mask.to(config.DEVICE)
        fake = model(degraded, delta, mask)
        loss = criterion(fake, clean)*config.L1_LAMBDA + ssim(fake, clean)

        opt.zero_grad(); loss.backward(); opt.step()

        loop.set_postfix(
            LOSS=loss.item(),
            epoch=epoch,
        )
    return model, opt

def main():
    trn_loader = train_loader()
    model, opt = init_Generator()


    criterion = L1()
    ssim = SSIMLoss()

    for epoch in range(config.NUM_EPOCHS):
        model, opt = train(model, opt, trn_loader, criterion, ssim, epoch)
        save_model(model, opt, config.URN_checkpoints)

def save_some_examples():
    loader = test_loader()
    model, _ = init_Generator()
    
    clean, degraded, delta, mask = next(iter(loader))
    clean, degraded = clean.to(config.DEVICE), degraded.to(config.DEVICE)
    delta, mask = delta.to(config.DEVICE), mask.to(config.DEVICE)

    model.eval()
    with torch.no_grad():
        fake = model(degraded, delta, mask)
        concat_cover = torch.cat((clean*.5+.5, degraded*.5+.5, fake*.5+.5), 2)
        save_image(concat_cover, f"outcomes/URN/output.png")

if __name__ == "__main__":
    main()
    save_some_examples()