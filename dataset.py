import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as tvt 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F 


# Mapping Pokemon type names to class labels 
TYPE_TO_LABEL = {
    'Normal': 0, 
    'Fire': 1, 
    'Water': 2, 
    'Grass': 3, 
    'Flying': 4, 
    'Fighting': 5, 
    'Poison': 6, 
    'Electric': 7, 
    'Ground': 8, 
    'Rock': 9, 
    'Psychic': 10, 
    'Ice': 11, 
    'Bug': 12, 
    'Ghost': 13, 
    'Steel': 14, 
    'Dragon': 15, 
    'Dark': 16, 
    'Fairy': 17
}

# Mapping Pokemon class labels to type names
LABEL_TO_TYPE = {v:k for (k,v) in TYPE_TO_LABEL.items()}

NUM_TYPES = len(TYPE_TO_LABEL.keys())

def pil_loader(path: str) -> Image:
    '''
    Load image using PIL. 
    Transparent portions should be white.

    Parameters
    ----------
    path: str
        path to image file

    Returns
    -------
    PIL Image: RGB loaded PIL image
    ''' 
    with Image.open(path) as im:
        im = im.convert('RGB')
    return im


def type_to_class_encoding(type_label: str) -> torch.Tensor:
    '''
    Converts pokemon type given in training CSV to an encoded tensor of size (num_classes, ) to be used by the network. For example, a type_label of "Electric/Ghost" would have a class encoding of [-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1].

    Parameters
    ----------
    type_label: str
        Pokemon type, with multiple types seperated by /

    Returns
    -------
    torch.Tensor: class encoding 
    '''
    types = type_label.split('/')
    labels = [TYPE_TO_LABEL[typ] for typ in types]
    label_encoding = -torch.ones(NUM_TYPES, dtype=torch.float32)
    label_encoding[labels] = 1
    return label_encoding


class PokemonSpriteDataset(Dataset):

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
        self.sprite_labels = [line.strip().split(',')[3] for line in lines]

        # Data augmentation transformation
        self.transform = tvt.Compose([
            tvt.Resize(96, tvt.InterpolationMode.NEAREST),
            # tvt.ColorJitter(0.4, 0.4, 0.4, 0.3),
            tvt.RandomHorizontalFlip(p=0.5),
            tvt.RandomAffine(30, translate=(0,0.1), scale=(0.9,1.1),fill=255),
            tvt.ToTensor(),
            tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return len(self.sprite_paths)

    def __getitem__(self, idx: int):
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
        img = pil_loader(self.sprite_paths[idx])
        img = self.transform(img)
        typ = self.sprite_labels[idx]
        label = type_to_class_encoding(typ)

        # img2 = 127.5*( 1+ np.array(img))
        # img2 = np.swapaxes(img2, 0, -1)
        # cv2.imwrite('test_poop.png', img2)
        
        return img, label 

def setup_dataloader(batch_size, root_dir='data', csv_file='sprites_metadata.csv'):
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
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )
    return dataloader
