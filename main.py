import torch
import torchvision.transforms as tvt 
from torch.utils.data import DataLoader

def setup_dataloader():
    '''
    Set up dataloader for training the pokemon sprite generator.

    Training dataset augmentations include 
    '''
    transform = tvt.Compose([
            tvt.RandomAffine(0, translate=(5/96, 5/96), fillcolor=(255,255,255)),
            tvt.ColorJitter(hue=0.5),
            tvt.RandomHorizontalFlip(p=0.5),
            tvt.ToTensor(),
            tvt.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
            ])
    dataset = ImageFolder(
            root=root,
            transform=transform
            )
    dataloader = DataLoader(dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            drop_last=True
            )