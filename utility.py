import torch
import dataset.config as config
import torch.optim as optim
from models.task_encoder import Text_Encoder
from models.modalityEncoder import Modality_Encoder
from models.generator import URN
from torchvision.utils import save_image
import torch.nn.functional as F



def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
def save_model(model, opt, path):
    save_checkpoint(model, opt, filename=path)

def init_txt_Model():
    model = Text_Encoder(in_dim=312).to(config.DEVICE)
    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    
    if config.LOAD_checkpoints_TEXT_Encoder:
        load_checkpoint(config.TEXT_MODEL_checkpoints, model, opt, config.LEARNING_RATE)
    return model, opt

def init_img_Model():
    model = Modality_Encoder(num_classes=4).to(config.DEVICE)
    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    
    if config.LOAD_checkpoints_modality_Encoder:
        load_checkpoint(config.IMAGE_ENCODER_checkpoints, model, opt, config.LEARNING_RATE)
    return model, opt
#================================================================================================
def init_Generator():
    model = URN().to(config.DEVICE)
    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    if config.GENERATOR_LOAD_checkpoints:
        load_checkpoint(config.URN_checkpoints, model, opt, config.LEARNING_RATE)
    return model, opt

def get_tasks_representation(img_enc, txt_enc, token, x, instruction, Points):
    with torch.no_grad():
        img_features = img_enc(x)
        txt_features = txt_enc(token(instruction))
    
    x_exp = img_features.unsqueeze(1) # (8, 1, 512)
    p_exp = Points.unsqueeze(0)       # (1, 5, 512)
    t_exp = txt_features.unsqueeze(1) # (8, 1, 512)
    
    dists = torch.norm(x_exp - p_exp, dim=2)  # (8, 5)
    similarities = 1 - F.cosine_similarity(t_exp, p_exp, dim=2)  # (8, 5)
    final = dists + similarities
    final_lbl = final.argmin(dim=1)
    # final_lbl = torch.tensor([3, 3, 2, 2, 1, 1, 4, 4], device='cuda:0')
    # print(final_lbl)
    closest_p = Points[final_lbl]
    return closest_p