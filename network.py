'''
Networks for the Pokemon sprite generator.

References:

'''
import os
from glob import glob
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from dataset import NUM_TYPES


class SpriteGAN(nn.Module):

    def __init__(self, lr: float, batch_size: int, ngf: int = 32, ndf: int = 32, latent_dim: int = 16):
        '''
        Pokemon sprite generator. 
        A combination of a conditional variational autoencoder (for stability and attribute control) and a GAN (for generation power).

        Parameters
        ----------
        lr: float
            learning rate
        ngf: int
            number of base filters to be used in the generator (encoder and decoder networks)
        ndf: int
            number of base filters to be used in the discriminators (ImageDiscriminator and LatentDiscriminator)
        latent_dim: int
            size of latent dimension
        isTrain: bool
            set to True if we want to train the network
        '''
        super(SpriteGAN, self).__init__()
        self.batch_size = batch_size
        self.latent_dim = latent_dim 

        # Networks
        self.encoder = Encoder(ngf, latent_dim)
        self.decoder = Decoder(ngf, latent_dim)
        self.disc_image = DiscriminatorImage(ndf, latent_dim)
        self.disc_latent = DiscriminatorLatent(ndf, latent_dim)

        # Optimizers
        self.opt_generator = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr)
        self.opt_disc_image = torch.optim.Adam(self.disc_image.parameters(), lr)
        self.opt_disc_latent = torch.optim.Adam(self.disc_latent.parameters(), lr)

        # Losses 
        self.real_label = torch.ones(batch_size).cuda()
        self.fake_label = torch.zeros(batch_size).cuda()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):

        # Generator loss
        z = self.encoder(x)
        x_recon = self.decoder(z, y)
        # conf_image = self.disc_image(x_recon, y)
        # conf_latent = self.disc_latent(z)

        recon_loss = F.l1_loss(x_recon, x)
        # gan_loss = F.binary_cross_entropy_with_logits(conf_image, self.real_label) + F.binary_cross_entropy_with_logits(conf_latent, self.real_label)
        # gen_loss = recon_loss + 1e-2 * gan_loss
        # gen_loss.backward()
        self.opt_generator.zero_grad()
        recon_loss.backward() # REMOVE
        self.opt_generator.step()

        # # Latent Discriminator loss
        # z_prior = 2 * torch.rand(self.batch_size, self.latent_dim) - 1
        # z_prior = z_prior.cuda()
        # conf_z_prior = self.disc_latent(z_prior)
        # conf_z = self.disc_latent(z.detach())

        # disc_latent_loss = F.binary_cross_entropy_with_logits(conf_z_prior, self.real_label) + F.binary_cross_entropy_with_logits(conf_z, self.fake_label)
        # self.opt_disc_latent.zero_grad()
        # disc_latent_loss.backward()
        # self.opt_disc_latent.step()

        # # Image Discriminator loss
        # conf_x = self.disc_image(x, y)
        # conf_x_recon = self.disc_image(x_recon.detach(), y)

        # disc_image_loss = F.binary_cross_entropy_with_logits(conf_x, self.real_label) + F.binary_cross_entropy_with_logits(conf_x_recon, self.fake_label)
        # self.opt_disc_image.zero_grad()
        # disc_image_loss.backward()
        # self.opt_disc_image.step()

        # Return losses
        loss_dict = {
            'generator/reconstruction_loss': recon_loss,
            # 'generator/gan_loss': gan_loss,
            # 'generator/total_loss': gen_loss,
            # 'discriminator/latent_loss': disc_latent_loss,
            # 'discriminator/image_loss': disc_image_loss
        }
        return loss_dict
    
    def sample(self, y: torch.Tensor):
        '''
        Sample a pokemon image with types <y>.

        Parameters
        ----------
        y: torch.Tensor
            one-hot encoding of pokemon types
        '''
        bs = y.shape[0]
        z = 2 * torch.rand(bs, self.latent_dim) - 1
        z = z.cuda()
        x = self.decoder(z, y)
        return x 
    

    def reconstruct(self, x: torch.Tensor, y: torch.Tensor):
        z = self.encoder(x)
        x_recon = self.decoder(z, y)
        return x_recon

    
    def save(self, save_dir: str, epoch: int) -> None:
        '''
        Save network weights.

        Parameters
        ----------
        save_dir: str
            path to save network weights
        epoch: int
            current epoch
        '''
        # Save
        torch.save(self.encoder.state_dict(), os.path.join(save_dir, "%i_enc.pth" % epoch))
        torch.save(self.decoder.state_dict(), os.path.join(save_dir, "%i_dec.pth" % epoch))
        torch.save(self.disc_image.state_dict(), os.path.join(save_dir, "%i_disc_image.pth" % epoch))
        torch.save(self.disc_latent.state_dict(), os.path.join(save_dir, "%i_disc_latent.pth" % epoch))

        # Only keep three most recent saves of our four models
        num_keep = 3 * 4
        fs = glob(os.path.join(save_dir, '*.pth'))
        fs.sort(key=os.path.getmtime)
        for f in fs[:-num_keep]:
            os.remove(f)

    
    def load(self, load_dir: str) -> None:
        '''
        Load network weights.

        Parameters
        ----------
        load_dir: str
            path to load network weights from
        '''
        if not load_dir:
            return

        # Find most recent epoch
        fs = glob(os.path.join(load_dir, '*.pth'))
        fs.sort(key=os.path.getmtime)
        epoch = int(fs[-1].split('_')[0])

        # Load
        self.encoder.load_state_dict(torch.load(os.path.join(load_dir, "%i_enc.pth" % epoch)))
        self.decoder.load_state_dict(torch.load(os.path.join(load_dir, "%i_dec.pth" % epoch)))
        self.disc_image.load_state_dict(torch.load(os.path.join(load_dir, "%i_disc_image.pth" % epoch)))
        self.disc_latent.load_state_dict(torch.load(os.path.join(load_dir, "%i_disc_latent.pth" % epoch)))



class Encoder(nn.Module):

    def __init__(self, num_filters: int = 32, latent_dim: int = 10):
        '''
        The encoder maps from the image space to the latent space.

        Parameters
        ----------
        num_filters: int
            base number of filters used, by default 32
        latent_dim: int
            size of latent dimension, by default 10
        '''
        super(Encoder,self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3,num_filters,5,2,2),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters,2*num_filters,5,2,2),
            nn.LeakyReLU(),
            nn.Conv2d(2*num_filters,4*num_filters,5,2,2),
            nn.LeakyReLU(),
            nn.Conv2d(4*num_filters,8*num_filters,5,2,2),
            nn.LeakyReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(9216,latent_dim), # Hard-cided
            # nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of encoder.

        Parameters
        ----------
        x: torch.Tensor
            input image, a tensor of shape (?,3,h,w)
        
        Returns
        -------
        torch.Tensor: latent vector of shape (?, latent_dim)
        '''
        # TODO - incorporate class here as well?
        conv = self.layers(x)
        conv = conv.view(-1,9216)
        out = self.fc(conv)
        return out



class Decoder(nn.Module):

    def __init__(self, num_filters: int = 32, latent_dim: int = 10, color_dim: int = 16):
        '''
        The decoder maps from the latent space to the image space.

        Parameters
        ----------
        num_filters: int
            base number of filters used, by default 32
        latent_dim: int
            size of latent dimension, by default 10
        '''
        super(Decoder,self).__init__()
        self.num_filters = num_filters
        self.color_dim = color_dim 

        def color_picker(input_dim: int, output_dim: int):
            '''
            Create and choose from a color palette during sprite generation.
            Helps with overall perceptual quality of network. 
            '''
            colorspace = nn.Sequential(
                nn.Linear(input_dim,128,bias=True),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128,64,bias=True),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Linear(64,output_dim,bias=True),
                nn.Tanh(),
            )
            return colorspace

        self.fc = nn.Sequential(
            nn.Linear(latent_dim+NUM_TYPES, 16*num_filters),
            nn.LeakyReLU()
        )
        self.upconv= nn.Sequential(
            nn.ConvTranspose2d(16*num_filters,8*num_filters,4,2,1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8*num_filters,4*num_filters,4,2,1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(4*num_filters,2*num_filters,4,2,1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(2*num_filters,num_filters,4,2,1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(num_filters,3,3,1,1), # TODO
            nn.Tanh() # nn.Softmax(),TODO
        )

        self.color_picker_r = color_picker(16*num_filters, color_dim)
        self.color_picker_g = color_picker(16*num_filters, color_dim)
        self.color_picker_b = color_picker(16*num_filters, color_dim)
        self.colourspace_upscaler = nn.Upsample(scale_factor=96)
        

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of generator network.

        Parameters
        ----------
        z: torch.Tensor
            latent input tensor, random noise
        y: torch.Tensor
            one-hot encoded type tensor
        
        Returns
        -------
        torch.Tensor: generated images
        '''
        # Generate 16-channel intermediate image from latent vector
        x = torch.cat([z, y],dim=1)
        x = self.fc(x)
        x_square = x.view(-1,16*self.num_filters,1,1)
        x_square = F.upsample(x_square, scale_factor=6)
        intermediate = self.upconv(x_square)
        return intermediate
        # Pick from color palette
        # r = self.color_picker_r(x)
        # r = r.view((-1, 16, 1, 1))
        # r = F.upsample(r, scale_factor=96)
        # r = intermediate * r
        # r = torch.sum(r, dim=1, keepdim=True) 

        # g = self.color_picker_g(x)
        # g = g.view((-1, self.color_dim, 1, 1))
        # g = F.upsample(g, scale_factor=96)
        # g = intermediate * g
        # g = torch.sum(g, dim=1, keepdim=True) 

        # b = self.color_picker_b(x)
        # b = b.view((-1, self.color_dim, 1, 1))
        # b = F.upsample(b, scale_factor=96)
        # b = intermediate * b
        # b = torch.sum(b, dim=1, keepdim=True) 

        # out = torch.cat((r, g, b), dim=1)
        # return out


class DiscriminatorImage(nn.Module):

    def __init__(self, num_filters: int = 32, latent_dim: int = 10):
        '''
        Discriminator for generated/real images.

        Parameters
        ----------
        num_filters: int 
            base number of filters used, by default 32
        '''
        super(DiscriminatorImage,self).__init__()
        self.num_filters = num_filters

        self.conv_img = nn.Sequential(
            nn.Conv2d(3,num_filters,4,2,1),
        )
        self.conv_l = nn.Sequential(
            nn.ConvTranspose2d(NUM_TYPES, NUM_TYPES, 48, 1, 0),
            nn.LeakyReLU()
        )
        self.total_conv = nn.Sequential(
            nn.Conv2d(num_filters+NUM_TYPES,num_filters*2,4,2,1),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters*2,num_filters*4,4,2,1),
            nn.LeakyReLU(),
            nn.Conv2d(num_filters*4,num_filters*8,4,2,1),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8*6*6*num_filters,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of image discriminator.

        Parameters
        ----------
        x: torch.Tensor
            input image, a tensor of shape (?, 3, h, w)
        y: torch.Tensor
            input latent, a tensor of shape (?, latent_dim)
        
        Returns
        -------
        torch.Tensor: real/fake activations, a vector of shape (?,)
        '''
        conv_img = self.conv_img(x)
        conv_l = self.conv_l(y.unsqueeze(-1).unsqueeze(-1))
        catted = torch.cat((conv_img,conv_l),dim=1)
        for layer in self.total_conv:
            catted = layer(catted)
        catted = catted.view(-1,8*6*6*self.num_filters)
        out = self.fc(catted)
        return out.squeeze()


class DiscriminatorLatent(nn.Module):

    def __init__(self, num_filters: int = 32, latent_dim: int = 10):
        '''
        Discriminator for latent vectors.

        Parameters
        ----------
        num_filters: int
            base number of filters used, by default 32
        latent_dim: int
            size of latent dimension, by default 10
        '''
        super(DiscriminatorLatent,self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(latent_dim,num_filters*4),
            nn.LeakyReLU(),
            nn.Linear(num_filters*4,num_filters*2),
            nn.LeakyReLU(),
            nn.Linear(num_filters*2,num_filters),
            nn.LeakyReLU(),
            nn.Linear(num_filters,1)
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of latent discriminator.

        Parameters
        ----------
        z: torch.Tensor
            input latent, a tensor of shape (?, latent_dim)
        
        Returns
        -------
        torch.Tensor: real/fake activations, a vector of shape (?,)
        '''
        return self.layers(z).squeeze()
