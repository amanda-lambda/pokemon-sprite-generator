from PIL import Image
import torch
import torchvision.transforms as tvt 
from torch.utils.data import DataLoader

def pil_loader(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        return im 


class PokemonSpriteDataset(torch.utils.Dataset):

    def __init__(self, csv_file='sprites_metadata.csv', root_dir='data'):
        '''
        Custom torch training dataset for conditional pokemon sprite generation.
        Training augmentations include 

        Params
        ------
        csv_file: str 
            Path to the csv file with annotations.
        root_dir: str
            Directory with all the images.
        '''
        # Extract information from metadata file
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        self.sprite_paths = [os.path.join(root_dir, line.split(',')[2]) for line in lines]
        self.sprite_types = [line.split(',')[3] for line in lines]

        # Data augmentation transformation
        self.transform = tvt.Compose([
            tvt.ColorJitter(0.4, 0.4, 0.4, 0.3),
            tvt.RandomHorizontalFlip(p=0.5),
            tvt.RandomAffine(30, translate=(0,0.1), scale=(0.9,1.1),fill=0)
            tvt.ToTensor(),
            # tvt.Normalize((0.5, 0.5, 0.5,), (0.5, 0.5, 0.5,))
        ])

    def __getitem__(self, idx):
        '''
        Grab item from the dataset.

        Params
        ------
        idx: int
            index of dataset item to grab
        
        Returns
        -------
        torch.Tensor: sprite in tensor of shape (C, H, W)
        torch.Tensor: one-hot encoded label
        '''
        img = pil_loader(self.sprite_paths[i])
        img = self.transform(img)
        label = self.sprite_labels[i]
        
        return img, label 

def setup_dataloader(batch_size, csv_file='sprites_metadata.csv', root_dir='data'):
    '''
    Set up dataloader for training the pokemon sprite generator.

    Params
    ------
    batch_size: int
        training batch size
    csv_file: str 
        Path to the csv file with annotations.
    root_dir: str
        Directory with all the images.
    
    Returns
    -------
    torch.utils.DataLoader: training dataloader
    '''
    dataset = PokemonSpriteDataset(csv_file, root_dir)
    dataloader = torch.utils.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            drop_last=False
    )
    return dataloader