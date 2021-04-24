# Pokemon Sprite Generator

Generate a pokemon sprite, given specified classes.

The network is a hybrid between VAE-GAN, AE-GAN, and a CVAE network in order to accomplish conditional image generation (see references below). One of the attractive things about AE-GAN is its training stability and robustness; I've had a lot of trouble in the past getting GANs to train stably, so this was a hugely appealing factor. Another aspect was the network's "ColorPicker", which acted as the final transposed convolutional layer in the network. It's output has a channel depth of 16 with a softmax activation applied to the channel dimension. This forced the generator to pick one of 16 indices for each pixel. Elsewhere in the network, the generator generated 16 colours (essentially a palette), and this palette was multiplied against the 16-depth index layer. This was done three times separately to create red, green, and blue channels for output, and allowed colours to be easily coordinated across the output image. I decided to incorporate the variational aspects of VAE-GAN and conditional aspects of CVAE so I could get more controllable image generation. i.e., instead of just being able to generate a Pokemon, I'd be able to generate a Pokemon based on what types I specified.

[NOTE] Work in progress! Under construction...

# Download the pokemon sprite data

Download the data:
```
cd pokemon-sprite-generator/
git clone https://github.com/PokeAPI/sprites.git
mv sprites/sprites/pokemon/*.png data/
rm -rf sprites
```

The associated type metadata is already provided at `sprites_metadata.csv`.

# Generate a pokemon

1. Download the pretrained model:
```
# TBD
wget <url>
```
Available types: 
```
python main.py --mode test 
```

# Train 

```
python main.py --mode train

# For full list of command line options:
python main.py -h 

usage: main.py [-h] [--mode {train,sample}] [--save_dir SAVE_DIR]
               [--load_dir LOAD_DIR] [--root_dir ROOT_DIR]
               [--csv_file CSV_FILE] [--batch_size BATCH_SIZE]
               [--learning_rate LEARNING_RATE] [--num_epochs NUM_EPOCHS]
               [--use_gpu]

pokemon-sprite-generator options

optional arguments:
  -h, --help            show this help message and exit
  --mode {train,sample}
                        run the network in train or sample mode
  --save_dir SAVE_DIR   path to save model and logs
  --load_dir LOAD_DIR   name of model to load
  --root_dir ROOT_DIR   path to the training data
  --csv_file CSV_FILE   path to the training data
  --batch_size BATCH_SIZE
                        batch size
  --learning_rate LEARNING_RATE
                        learning rate
  --num_epochs NUM_EPOCHS
                        number of epochs
  --use_gpu             if set, train on gpu instead of cpu
```

# Requirements
```
torch 
torchvision

```
# References

- https://github.com/ConorLazarou/PokeGAN