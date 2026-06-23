import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append('models')
sys.path.append('dataset')

import dataset.config as config
from tqdm import tqdm
import torch
# from dataset.img_dataset import train_loader
from dataset.txt_dataset import train_loader
from loss import TripletLoss, CosineTripletLoss
from utility import init_img_Model, init_txt_Model, save_model
from models.task_encoder import Tokenizer, DistilBERT_Tokenizer, TinyBERT_Tokenizer, MobileBERT_Tokenizer, MPNet_Tokenizer, multilingualBERT_Tokenizer

def Text_train(model, opt, token, criterion, loader, points, epoch):
    loop = tqdm(loader, leave=True)
    for idx, (d) in enumerate(loop):
        # labels = d['label']
        pos_text = d['pos_text']
        neg_text = d['neg_text']
        # point_idx = d['pos_idx']
        anc = d['anc'].to(config.DEVICE)

        p = token(pos_text)
        n = token(neg_text)

        with torch.cuda.amp.autocast():
            pos = model(p)
            neg = model(n)

            # loss = criterion(a, p, n)
            loss = criterion(anc, pos, neg)

        opt.zero_grad(); loss.backward(); opt.step()


        loop.set_postfix(
            LOSS=loss.item(),
            epoch=epoch,
        )
    return model, opt

def main_Text():
    trn_loader = train_loader()
    #-----------------------------------------------------------------
    model, opt = init_txt_Model()
    token = TinyBERT_Tokenizer(config.DEVICE) #Tokenizer(config.DEVICE)
    #-----------------------------------------------------------------
    points = config.get_Points()
    criterion = CosineTripletLoss()
    # save_model(model, opt)
    #-----------------------------------------------------------------

    for epoch in range(config.NUM_EPOCHS):
        model, opt = Text_train(model, opt, token, criterion, trn_loader, points, epoch)        
        save_model(model, opt, config.TEXT_MODEL_checkpoints)

if __name__ == "__main__":
    main_Text()