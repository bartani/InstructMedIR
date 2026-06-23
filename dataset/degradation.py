import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance

#tasks = ['low-dose', 'noise', 'SR', 'blur']

def task_degradation(x, task_index):
    if task_index == 0: # for low-dose task
        x_deg = low_dose_degrade(x)
            
    elif task_index == 1: # for denoising task
        x_deg = add_noise(x, sigma_choices = [50])
            
    elif task_index == 2: # for RS task
        x_deg = super_resolution_degrade(x)
            
    elif task_index == 3: # for debulring task
        x_deg = add_blur(x)
        
    else:
        x_deg = x
    
    return x_deg

def add_noise(img, sigma_choices = [15, 25, 50]):
    img_tensor = TF.to_tensor(img)
    sigma = random.choice(sigma_choices)
    sigma = sigma / 255.0
    noise = torch.randn_like(img_tensor) * sigma
    noisy_img = torch.clamp(img_tensor + noise, 0.0, 1.0)
    return TF.to_pil_image(noisy_img)

def add_blur(img, radius_range=(1, 7)):
    radius = random.uniform(*radius_range)
    return img.filter(ImageFilter.GaussianBlur(radius))

def super_resolution_degrade(img, scale_factor=4):
    w, h = img.size
    img_low = img.resize(
        (w // scale_factor, h // scale_factor),
        Image.BILINEAR
    )
    img_up = img_low.resize((w, h), Image.BILINEAR)
    return img_up

def low_dose_degrade(img, brightness_range=(0.4, 0.7)):
    factor = random.uniform(*brightness_range)
    enhancer = ImageEnhance.Brightness(img)
    img_dim = enhancer.enhance(factor)

    # Add slight noise
    return add_noise(img_dim, sigma_choices = [5, 10])
    