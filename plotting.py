import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
sys.path.append('models')
sys.path.append('dataset')

from dataset.img_dataset import test_loader, train_loader
import dataset.config as config
from utility import init_img_Model, init_txt_Model

from models.task_encoder import Tokenizer, DistilBERT_Tokenizer, TinyBERT_Tokenizer, MPNet_Tokenizer, multilingualBERT_Tokenizer

from tqdm import tqdm
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision.datasets as datasets
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score





def extract_embedding(data):
    points = config.get_Points()
    model, _ = init_txt_Model()
    token = TinyBERT_Tokenizer(config.DEVICE)
    model.eval()
    #-----------------------------------------------------------------
    embeddings = []
    labels = []
    #-----------------------------------------------------------------
    for i in range(len(data)):
        row = data.iloc[i]   
        text = row['instruction']
        lbl = torch.tensor([config.get_class_idx(row['class'])])

        
        # with torch.no_grad():
        #     t = token(text)
        #     t = t.view(t.shape[0], -1)
        #     embeddings.append(t.cpu())
        #     labels.append(lbl)


        with torch.no_grad():
            t = token(text)
            t = model(t)
            embeddings.append(t.cpu())
            labels.append(lbl)
    
    
    embeddings.append(points)
    for i in range(len(points)):
        labels.append(torch.tensor([i]))
    
    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    return embeddings, labels

def tSNE_text_Plot():
    points = config.get_Points()
    file_path = os.path.join(".", "data", "test_instruction-4D.csv")
    data = pd.read_csv(file_path)
    len_data = len(data)
    print(len_data)
    
    embeddings, labels = extract_embedding(data)

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d  = tsne.fit_transform(embeddings)
    

    plt.figure(figsize=(6, 5))
    palette = sns.color_palette("hsv", len(data['class'].unique()))

    for idx, class_name in enumerate(data['class'].unique()):
        class_idx = config.get_class_idx(class_name)
        indices = labels == class_idx
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1],
                    label=class_name, alpha=0.6, s=30, color=palette[class_idx])

    for i in range(len(points)):
        x, y = embeddings_2d[len_data+i, 0], embeddings_2d[len_data+i, 1]
        plt.scatter(x, y, color='black', marker='X', s=100, edgecolors='black')
        plt.text(x + 1.5, y, config.tasks[i], fontsize=12, weight='bold', color='black')

    plt.legend()
    # plt.title("t-SNE of Encoder Outputs")
    # plt.xlabel("t-SNE 1")
    # plt.ylabel("t-SNE 2")
    plt.tight_layout()
    plt.savefig("rawtSNE.png", dpi=400, bbox_inches='tight')
    plt.show()

def confusion_matrix_images():
    tst_loader = test_loader()
    #-----------------------------------------------------------------
    model, _ = init_img_Model()
    #-----------------------------------------------------------------
    class_names = {'MRI':0, 'CT':1, 'X-Ray':2, 'PET':3}
    plot_confusion_matrix(model, tst_loader, config.DEVICE, class_names)

def plot_confusion_matrix(model, loader, device, class_names):
    model.eval()
    
    all_preds = []
    all_labels = []

    
    loop = tqdm(loader, leave=True)
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loop):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            target = torch.argmax(labels, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # --------------------------
    # Plot (Publication Quality)
    # --------------------------
    # Normalize row-wise (recall-based)
    cm_normalized = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
    cm_percent = cm_normalized * 100

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm_percent, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(np.arange(len(class_names)), class_names, rotation=45, fontsize=11)
    plt.yticks(np.arange(len(class_names)), class_names, fontsize=11)

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    # plt.title("Confusion Matrix (%)", fontsize=13)

    # Add percentage values
    threshold = cm_percent.max() / 2.
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            value = cm_percent[i, j]
            plt.text(j, i, f"{value:.1f}%",
                     ha="center",
                     va="center",
                     color="white" if value > threshold else "black",
                     fontsize=10)

    plt.tight_layout()

    plt.savefig("confusion_matrix_miccai.png", dpi=400, bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    tSNE_text_Plot()
    # confusion_matrix_images()