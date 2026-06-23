import config
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random
from degradation import task_degradation

class train_dataset(Dataset):
    def __init__(self, imgs):
        self.files = imgs
        self.lendata = len(self.files)

    def __len__(self):
        return self.lendata*1

    def __getitem__(self, index):
        _path = self.files[index % self.lendata]
        modality_path = Path(_path)
        modality_name = modality_path.parent.name

        # generate mask
        mask = config.get_mask_based_on_modality_name(modality_name).float()
        x = Image.open(_path).convert('RGB')
        selected_task = random.choice(config.tasks) 
        x = task_degradation(x, config.tasks.index(selected_task))
       
        x = config.weak_tfms(x)

        return x, mask
    
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
    myds = train_dataset(config.get_image_files(config.TEST_PATH+"/*"))
    loader = DataLoader(
        myds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    return loader
    
if __name__ == "__main__":    
    loader = train_loader()
    
    x, lbl =next(iter(loader))
    print(x.shape)
    print(lbl.shape)
    print(lbl)

    # save_image(concat_cover, "outcomes/datasample/samplepos_neg.png")