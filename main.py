import os
import typing
import uuid
import argparse 
from glob import glob

import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as tvt 
from torchvision.utils import make_grid, save_image

from dataset import setup_dataloader, type_to_class_encoding
from network import SpriteGAN, Decoder



def train(root_dir: str, csv_file: str, 
          load_dir: str, save_dir: str, 
          num_epochs: int, batch_size: int, lr: float, 
          use_gpu: bool) -> None:
    '''
    Train the SpriteGAN.
    Saves checkpoints and training logs to `save_dir`.

    Parameters
    ----------
    root_dir: str
        root folder for training data
    csv_file: str
        sprites metadata training file, contains image paths and type labels
    load_dir: str
        folder to load network weights from
    save_dir: str
        folder to save network weights to
    num_epochs: int
        number of epochs to train the network
    batch_size: int
        batch size for each network update step
    lr: float
        learning rate for generator, discriminator will have learning rate lr/2
    use_gpu: bool
        Set to true to run training on GPU, otherwise run on CPU
    '''
    # Dataloader
    loader = setup_dataloader(batch_size, root_dir, csv_file)

    # Writer
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    writer = SummaryWriter(save_dir)

    # Network
    sprite_gan = SpriteGAN(lr, batch_size, use_gpu)
    if load_dir:
        sprite_gan.load(load_dir)

    # Train
    step = 0
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(loader):
            if use_gpu:
                x, y = x.cuda(), y.cuda()
            loss_dict = sprite_gan.forward(x, y)
            step += 1
            print("Step % i" % step, loss_dict)

            # Log
            if i % 100 == 0:
                for (k,v) in loss_dict.items():
                    writer.add_scalar(k, v.item(), step)
        
        # Save
        if epoch % 20 == 0:
            sprite_gan.save(save_dir, epoch)
            print("Save new model at epoch %i." % epoch)

            # Log sample image
            with torch.no_grad():
                n_sample = 3
                x_gen = sprite_gan.sample(y[:n_sample])
                gen_image = make_grid(x_gen, nrow=n_sample, normalize=True, value_range=(-1,1))
                writer.add_image('image/gen', gen_image, epoch)

                x_recon = sprite_gan.reconstruct(x[:n_sample], y[:n_sample])
                recon_image = make_grid(x_recon, nrow=n_sample, normalize=True, value_range=(-1,1))
                writer.add_image('image/recon', recon_image, epoch)

                orig_image = make_grid(x[:n_sample], nrow=n_sample, normalize=True, value_range=(-1,1))
                writer.add_image('image/orig', orig_image, epoch)


def sample(types: str, load_dir: str, save_dir: str, use_gpu: bool) -> None:
    '''
    Sample the SpriteGAN.

    Parameters
    ----------
    types: List
        types, seperated by /, e.g. 'Poison/Steel'
    load_dir: str
        folder to load network weights from
    save_dir: str
        folder to save network weights to
    use_gpu: bool
        Set to true to run training on GPU, otherwise run on CPU
    '''
    # Network
    model = Decoder()
    model = model.eval()
    if use_gpu:
        model = model.cuda()
    if load_dir:
        fs = glob(os.path.join(load_dir, '*_dec.pth'))
        fs.sort(key=os.path.getmtime)
        model.load_state_dict(torch.load(fs[-1]))

    # Types list to one-hot encoded label vector
    y = type_to_class_encoding(types)
    y = y.unsqueeze(0)
    
    # Generate
    z = torch.randn((1, model.latent_dim))
    x = model.forward(z, y)

    # Save
    save_path = os.path.join(save_dir, str(uuid.uuid1()) + '.png')
    save_image(x.squeeze(), save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="pokemon-sprite-generator options")

    # MODE options
    parser.add_argument("--mode",
                        type=str,
                        help="run the network in train or sample mode",
                        default="train",
                        choices=["train", "sample"])

    # ALL options (train or sample)
    parser.add_argument("--save_dir",
                        type=str,
                        help="path to save model, logs, generated images",
                        default="logs")
    parser.add_argument("--load_dir",
                        type=str,
                        help="path to model to load",
                        default="")
    parser.add_argument("--use_gpu",
                        help="if set, train on gpu instead of cpu",
                        action="store_true")

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
                        default=32)
    parser.add_argument("--learning_rate",
                        type=float,
                        help="learning rate",
                        default=2e-4)
    parser.add_argument("--num_epochs",
                        type=int,
                        help="number of epochs",
                        default=50000)
    
    # TESTING options
    parser.add_argument("--types",
                        type=str,
                        help="pokemon types, comma seperated")

    
    # Parse arguments! Run train or sample.
    options = parser.parse_args()
    if options.mode == 'train':
        train(options.root_dir, options.csv_file, options.load_dir, options.save_dir, options.num_epochs, options.batch_size, options.learning_rate, options.use_gpu)
    elif options.mode == 'sample':
        sample(options.types, options.load_dir, options.save_dir, options.use_gpu)
