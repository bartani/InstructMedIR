import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import glob
import random
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#-----------------------------------------------------------------
LOAD_checkpoints_modality_Encoder = True
IMAGE_ENCODER_checkpoints = f"checkpoints/modality_enc_4D.pth.tar"
#-----------------------------------------------------------------
TEXT_MODEL_checkpoints = f"checkpoints/TinyBERT_4D.pth.tar" 
LOAD_checkpoints_TEXT_Encoder = True
#-----------------------------------------------------------------
GENERATOR_LOAD_checkpoints = True
URN_checkpoints = "checkpoints/URN.pth.tar"
#-----------------------------------------------------------------
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
L1_LAMBDA = 100
#--------------------------------------------------------------------------
TRAIN_PTH = "modality dataset/train"
TEST_PATH = "modality dataset/test"
#--------------------------------------------------------------------------
ZOOM_SIZE = 300
IMAGE_SIZE = 256
batch_size = 1
num_workers = 2
K = 4
modalities = ["MRI", "CT", "X-Ray", "PET"]
tasks = ['low-dose', 'noise', 'SR', 'blur']
#--------------------------------------------------------------------------
def get_mask_based_on_modality_name(modality_name):
    modality_to_idx = {m: i for i, m in enumerate(modalities)}
    index = modality_to_idx[modality_name]
    mask = torch.nn.functional.one_hot(torch.tensor(index), num_classes=len(modalities))
    return mask




def get_class_idx(class_name):
    task_idx = tasks.index(class_name)
    return task_idx

def get_Points():
    file_path = os.path.join(".", "data", "4_points.pt")
    loaded_points = torch.load(file_path)
    return loaded_points

def get_class_points(points, labels):
    task_to_idx = {task: idx for idx, task in enumerate(tasks)}
    label_indices = [task_to_idx[label] for label in labels]
    label_indices = torch.tensor(label_indices)
    selected_points = points[label_indices]
    
    return selected_points, label_indices
#-----------------------------------------------------------------
tfms = transforms.Compose([       # Tensor
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   
])
test_tfms = transforms.Compose([       # Tensor
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   
])
weak_tfms = transforms.Compose([
    transforms.Resize((ZOOM_SIZE,ZOOM_SIZE)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10, expand=False),
    transforms.RandomVerticalFlip(0.5),
    transforms.ColorJitter(brightness=.5, contrast=.1, saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
#-----------------------------------------------------------------
def get_image_files(folder_path):
    # Define the allowed image file extensions
    allowed_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]  # Add more if needed

    image_files = []
    for ext in allowed_extensions:
        # Use glob to find files with allowed extensions in the folder path
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    return image_files

def apply_transforms(img1, img2):

    if random.random() > 0.5:
        img1 = TF.vflip(img1)
        img2 = TF.vflip(img2)

    if random.random() > 0.5:
        img1 = TF.hflip(img1)
        img2 = TF.hflip(img2)

        
    if random.random() > 0.5:
        angle = random.random()*15
        img1 = TF.rotate(img1, angle=angle)
        img2 = TF.rotate(img2, angle=angle)
    
    return img1, img2
#-----------------------------------------------------------------