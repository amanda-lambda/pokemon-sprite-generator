import os 
import argparse 

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as tvt 
from dataset import setup_dataloader


def train(root_dir: str, csv_file: str, 
          load_dir: str, save_dir: str, 
          batch_size: int, lr: float, 
          use_gpu: bool):
    # Dataloader
    loader = setup_dataloader(batch_size, root_dir, csv_file)

    # Writer
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    writer = SummaryWriter(save_dir)

    # Network
    from network import SpriteGAN
    sprite_gan = SpriteGAN(lr, batch_size)
    if use_gpu:
        sprite_gan = sprite_gan.cuda()

    # Train
    step = 0
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(loader):
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            loss_dict = sprite_gan.forward(x, y)
            step += 1

            # Log
            if i % 100 == 0:
                for (k,v) in loss_dict:
                    writer.add_scalar(k, v.item(), step)
        
        # Save
        if epoch % 10 == 0:
            sprite_gan.save(save_dir, epoch)

            # Log sample image
            with torch.no_grad():
                x = sprite_gan.sample(y)
                val_image = make_grid(x, nrow=1)
                self.writer.add_image('image', val_image, epoch)

def test(root_dir, csv_file, load_dir):
    return

if name == '__main__':
    parser = argparse.ArgumentParser(description="pokemon-sprite-generator options")
    parser.add_argument("--mode",
                        type=str,
                        help="run the network in train or sample mode",
                        default="train",
                        choices=["train", "sample"])

    # DIRECTORY options
    parser.add_argument("--save_dir",
                        type=str,
                        help="path to save model and logs",
                        default="logs")
    parser.add_argument("--load_dir",
                        type=str,
                        help="name of model to load")

    # TRAINING options
    parser.add_argument("--root_dir",
                        type=str,
                        help="path to the training data",
                        default="data")
    parser.add_argument("--csv_file",
                        type=str,
                        help="path to the training data",
                        default="sprites_metadata.csv")
    parser.add_argument("--batch_size",
                        type=int,
                        help="batch size",
                        default=16)
    parser.add_argument("--learning_rate",
                        type=float,
                        help="learning rate",
                        default=2e-4)
    parser.add_argument("--num_epochs",
                        type=int,
                        help="number of epochs",
                        default=100)
    parser.add_argument("--use_gpu",
                        help="if set, train on gpu instead of cpu",
                        action="store_true")

    options = parser.parse_args()
    if options.mode == 'train':
        train(options.root_dir, options.csv_file, options.load_dir, options.save_dir, options.batch_size, options.lr, options.use_gpu)
    elif options.mode == 'sample':
        sample()