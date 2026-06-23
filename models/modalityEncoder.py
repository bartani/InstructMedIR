from torchvision import models
import torch.nn as nn
import torch
import torch.nn.functional as F



class Modality_Encoder(nn.Module):
    def __init__(self, num_classes=4):
        super(Modality_Encoder, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):   
        x = F.interpolate(
            x,
            size=(224, 224),
            mode='bilinear',
            align_corners=False
        )     
        return self.model(x)

if __name__ == "__main__":
    x = torch.randn((4, 3, 224, 224))
    model = Modality_Encoder(num_classes=4)
    pred = model(x)
    print(pred.shape)