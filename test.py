import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append('models')
sys.path.append('dataset')

from models.task_encoder import TinyBERT_Tokenizer
from utility import init_img_Model, init_txt_Model, init_Generator
import dataset.config as config
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import save_image


def enhance_modality(path, instruction, save_path):
    x = Image.open(path).convert('RGB')
    x = config.tfms(x)
    x = x.unsqueeze(0).to(config.DEVICE) 
    # print(x.shape)

    img_enc, _ = init_img_Model()
    pred = img_enc(x)
    pred = torch.argmax(pred, dim=1)
    mask = torch.nn.functional.one_hot(torch.tensor(pred), num_classes=len(config.modalities))
    # print("pred:", pred)
    # print("mask:", mask)

    txt_enc, _ = init_txt_Model()
    token = TinyBERT_Tokenizer(config.DEVICE)

    text = []
    text.append(instruction)
    task = txt_enc(token(text))
    # print(task.shape)
    points = config.get_Points().to(config.DEVICE)
    # print("points", points.shape)
    sim = F.cosine_similarity(task, points, dim=1)
    # print(sim)
    target_task = sim.argmax()
    delta = points[target_task].unsqueeze(0) 
    # print(delta.shape)

    model, _ = init_Generator()
    enhanced = model(x, delta.float(), mask.float())
    save_image(enhanced*.5+.5, f"{save_path}enhanced.png")



if __name__ == "__main__":
    save_path = "outcomes/datasample/"
    path = "outcomes/datasample/sample1.png"
    instruction = "Can someone suppress structured noise in this MRI image affecting the small lung nodules while preserving natural texture for accurate diagnosis?"
    # instruction = "Please resolve out-of-plane motion blur affecting the abdominal organs without distorting fine structures ?"
    enhance_modality(path, instruction, save_path)
