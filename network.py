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
from torch.nn.utils import spectral_norm

from dataset import NUM_TYPES


class SpriteGAN(nn.Module):

    def __init__(self, lr: float, batch_size: int, ngf: int = 64, ndf: int = 64, latent_dim: int = 100):
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

        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)
        self.disc_image.apply(weights_init)
        self.disc_latent.apply(weights_init)

        # Optimizers
        self.opt_generator = torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr)
        self.opt_disc_image = torch.optim.Adam(self.disc_image.parameters(), lr, betas=(0.5, 0.9))
        self.opt_disc_latent = torch.optim.Adam(self.disc_latent.parameters(), lr, betas=(0.5, 0.9))

        # Losses 
        self.real_label = torch.ones(batch_size).cuda()
        self.fake_label = torch.zeros(batch_size).cuda()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):

        # Latent Discriminator loss
        self.opt_disc_latent.zero_grad()
        z = self.encoder(x)
        x_recon = self.decoder(z, y)

        z_prior = torch.randn((self.batch_size, self.latent_dim), requires_grad=True)
        z_prior = z_prior.cuda()
        conf_z_prior = self.disc_latent(z_prior)
        conf_z = self.disc_latent(z.detach())

        disc_latent_loss = F.binary_cross_entropy_with_logits(conf_z_prior, self.real_label) + F.binary_cross_entropy_with_logits(conf_z, self.fake_label)
        disc_latent_loss.backward()
        self.opt_disc_latent.step()

        # Image Discriminator loss
        self.opt_disc_image.zero_grad()
        conf_x = self.disc_image(x, y)
        conf_x_recon = self.disc_image(x_recon.detach(), y)

        disc_image_loss = F.binary_cross_entropy_with_logits(conf_x, self.real_label) + F.binary_cross_entropy_with_logits(conf_x_recon, self.fake_label)
        disc_image_loss.backward()
        self.opt_disc_image.step()

        # Generator loss
        self.opt_generator.zero_grad()
        z = self.encoder(x)
        x_recon = self.decoder(z, y)
        conf_image = self.disc_image(x_recon, y)
        conf_latent = self.disc_latent(z)

        recon_loss = F.l1_loss(x_recon, x)
        gan_loss = F.binary_cross_entropy_with_logits(conf_image, self.real_label) + F.binary_cross_entropy_with_logits(conf_latent, self.real_label)
        gen_loss = recon_loss + 0.1 * gan_loss
        gen_loss.backward()
        self.opt_generator.step()

        # Return losses
        loss_dict = {
            'generator/reconstruction_loss': recon_loss,
            'generator/gan_loss': gan_loss,
            'generator/total_loss': gen_loss,
            'discriminator/latent_loss': disc_latent_loss,
            'discriminator/image_loss': disc_image_loss
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



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Encoder(nn.Module):

    def __init__(self, num_filters: int = 64, latent_dim: int = 100):
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
            # nn.BatchNorm2d(num_filters),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_filters,2*num_filters,5,2,2),
            # nn.BatchNorm2d(2*num_filters),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(2*num_filters,4*num_filters,5,2,2),
            # nn.BatchNorm2d(4*num_filters),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(4*num_filters,8*num_filters,5,2,2),
            # nn.BatchNorm2d(8*num_filters),
            nn.Dropout(p=0.3),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(18432,latent_dim)

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
        batch_size = x.shape[0]
        conv = self.layers(x)
        conv = conv.view(batch_size,-1)
        out = self.fc(conv)
        return out



class Decoder(nn.Module):

    def __init__(self, num_filters: int = 64, latent_dim: int = 100, color_dim: int = 16):
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
                nn.ReLU(True),
                nn.Linear(128,64,bias=True),
                nn.BatchNorm1d(64),
                nn.ReLU(True),
                nn.Linear(64,output_dim,bias=True),
                nn.Tanh(),
            )
            return colorspace

        self.fc = nn.Sequential(
            nn.Linear(latent_dim+NUM_TYPES, 16*num_filters),
            nn.BatchNorm1d(16*num_filters),
            nn.ReLU(True)
        )

        self.upconv= nn.Sequential( # 6 12 24 48 96
            nn.Conv2d(16*num_filters,8*num_filters,3,1,1),
            nn.BatchNorm2d(8*num_filters),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8*num_filters,4*num_filters,3,1,1),
            nn.BatchNorm2d(4*num_filters),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(4*num_filters,2*num_filters,3,1,1),
            nn.BatchNorm2d(2*num_filters),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(2*num_filters,num_filters,3,1,1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_filters,num_filters,3,1,1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(num_filters,color_dim,3,1,1), 
            nn.Softmax(),
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
        x_square = F.upsample(x_square, scale_factor=3)
        intermediate = self.upconv(x_square)

        # Pick from color palette
        r = self.color_picker_r(x)
        r = r.view((-1, 16, 1, 1))
        r = F.upsample(r, scale_factor=96)
        r = intermediate * r
        r = torch.sum(r, dim=1, keepdim=True) 

        g = self.color_picker_g(x)
        g = g.view((-1, self.color_dim, 1, 1))
        g = F.upsample(g, scale_factor=96)
        g = intermediate * g
        g = torch.sum(g, dim=1, keepdim=True) 

        b = self.color_picker_b(x)
        b = b.view((-1, self.color_dim, 1, 1))
        b = F.upsample(b, scale_factor=96)
        b = intermediate * b
        b = torch.sum(b, dim=1, keepdim=True) 

        out = torch.cat((r, g, b), dim=1)
        return out


class DiscriminatorImage(nn.Module):

    def __init__(self, num_filters: int = 64, latent_dim: int = 100):
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
            spectral_norm(nn.Conv2d(3,num_filters,4,2,1)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv_l = nn.Sequential(
            nn.ConvTranspose2d(NUM_TYPES, NUM_TYPES, 48, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.total_conv = nn.Sequential(
            spectral_norm(nn.Conv2d(num_filters+NUM_TYPES,num_filters*2,4,2,1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(num_filters*2,num_filters*4,4,2,1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(num_filters*4,num_filters*8,4,2,1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Sequential(
            spectral_norm(nn.Linear(8*6*6*num_filters,1024)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(1024,1))
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

    def __init__(self, num_filters: int = 64, latent_dim: int = 100):
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
            spectral_norm(nn.Linear(latent_dim,num_filters*4)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(num_filters*4,num_filters*2)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(num_filters*2,num_filters)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(num_filters,1))
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
