# Pokemon Sprite Generator

Generate a pokemon sprite, given specified classes.

The network is a hybrid between VAE-GAN, AE-GAN, and a CVAE network in order to accomplish conditional image generation (see references below). One of the attractive things about AE-GAN is its training stability and robustness; I've had a lot of trouble in the past getting GANs to train stably, so this was a hugely appealing factor. Another aspect was the network's "ColorPicker", which acted as the final transposed convolutional layer in the network. It's output has a channel depth of 16 with a softmax activation applied to the channel dimension. This forced the generator to pick one of 16 indices for each pixel. Elsewhere in the network, the generator generated 16 colours (essentially a palette), and this palette was multiplied against the 16-depth index layer. This was done three times separately to create red, green, and blue channels for output, and allowed colours to be easily coordinated across the output image. I decided to incorporate the variational aspects of VAE-GAN and conditional aspects of CVAE so I could get more controllable image generation. i.e., instead of just being able to generate a Pokemon, I'd be able to generate a Pokemon based on what types I specified.

# Download the pokemon sprite data

Download the data:
```
cd pokemon-sprite-generator/
git clone https://github.com/PokeAPI/sprites.git
mv sprites/sprites/pokemon/*.png data/
rm -rf sprites
```

# Train 

```
python main.py
```

# Generate a pokemon
```
python main.py 
```

# Requirements
```
torch 
torchvision

```
# References

- https://github.com/ConorLazarou/PokeGAN