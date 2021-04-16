'''
Networks for the Pokemon sprite generator.

References:

'''
import torch
from torch import nn
from torch.autograd import Variable

from dataset import NUM_TYPES


class SpriteGAN(nn.Module):

    def __init__(self, lr, ngf: int = 32, ndf: int = 32, latent_dim: int = 10):
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

        # Networks
        self.encoder = Encoder(ngf, latent_dim)
        self.decoder = Decoder(ngf, latent_dim)
        self.disc_image = DiscriminatorImage(ndf)
        self.disc_latent = DiscriminatorLatent(ndf, latent_dim)

        # Optimizers
        self.opt.generator = torch.optim.Adam(self.encoder.parameters() + self.decoder.parameters(), lr)
        self.opt_disc_image = torch.optim.Adam(self.disc_image.parameters(), lr)
        self.opt_disc_latent = torch.optim.Adam(self.disc_latent.parameters(), lr)

        # Losses 
        self.z_prior = Variable(torch.FloatTensor(batchSize*n_z).uniform_(-1,1)).view(batchSize,n_z)
        self.real_label = Variable(torch.ones(batchSize).fill_(1)).view(-1,1)
        self.fake_label = Variable(torch.ones(batchSize).fill_(0)).view(-1,1)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):

        # Generator loss
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        z = self.encoder(x)
        x_recon = self.decoder(z, y)
        conf_image = self.disc_image(x_recon, y)
        conf_latent = self.disc_latent(z)

        recon_loss = F.l1_loss(x_recon, x)
        gan_loss = F.inary_cross_entropy(conf_image, real_label)
        gen_loss = recon_loss + 1e-4 * gan_loss
        gen_loss.backward()
        self.opt_generator.step()

        # Latent Discriminator loss
        self.disc_latent.zero_grad()
        conf_z_prior = self.disc_latent(self.z_prior)
        conf_z = self.disc_latent(z.detach())

        disc_latent_loss = F.binary_cross_entropy(conf_z_prior, self.real_label)
        disc_latent_loss.backward()
        self.opt_disc_latent.step()

        # Image Discriminator loss
        self.disc_image.zero_grad()
        conf_x = self.disc_image(x, y)
        conf_x_recon = self.disc_image(x_recon.detach(), y)

        disc_image_loss = F.binary_cross_entropy(conf_x, self.real_label) + F.binary_cross_entropy(conf_x_recon, self.fake_label)
        disc_image_loss.backward()
        self.opt_disc_image.step()

        # Return losses
        loss_dict = {
            'generator/reconstruction_loss': recon_loss,
            'generator/gan_loss': gan_loss,
            'generator/total_loss': gen_loss,
            'discriminator/latent_loss': disc_latent_loss,
            'discriminator/image_loss': disc_image_Loss
        }
        return loss_dict



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
            nn.ReLU(),
            nn.Conv2d(num_filters,2*num_filters,5,2,2),
            nn.ReLU(),
            nn.Conv2d(2*num_filters,4*num_filters,5,2,2),
            nn.ReLU(),
            nn.Conv2d(4*num_filters,8*num_filters,5,2,2),
            nn.ReLU(),
        )
        self.fc = nn.Linear(8*num_filters*8*8,latent_dim)

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
        conv = self.layers(x).view(-1,8*num_filters*8*8)
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

        def color_picker(self, input_dim: int, output_dim: int):
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
            nn.Linear(latent_dim+NUM_TYPES, 8*8*num_filters*16),
            nn.ReLU()
        )
        self.upconv= nn.Sequential(
            nn.ConvTranspose2d(16*num_filters,8*num_filters,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(8*num_filters,4*num_filters,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(4*num_filters,2*num_filters,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(2*num_filters,num_filters,4,2,1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,3,3,1,1),
            nn.Softmax(),
        )

        self.color_picker_r = color_picker(8*8*num_filters*16, 16)
        self.color_picker_g = color_picker(8*8*num_filters*16, 16)
        self.color_picker_b = color_picker(8*8*num_filters*16, 16)
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
        x_square = x.view(-1,16*n_gen,8,8)
        intermediate = self.upconv(x_square)

        # Pick from color palette
        r = self.color_picker_r(x)
        r = r.view((-1, 16, 1, 1))
        r = F.upsample(r, scale_factor=96)
        r = intermediate * r

        g = self.color_picker_g(x)
        g = g.view((-1, 16, 1, 1))
        g = F.upsample(g, scale_factor=96)
        g = intermediate * g

        b = self.color_picker_b(x)
        b = b.view((-1, 16, 1, 1))
        b = F.upsample(b, scale_factor=96)
        b = intermediate * b

        out = torch.cat((r, g, b), dim=1)
        return out


class DiscriminatorImage(nn.Module):

    def __init__(self, num_filters: int = 32):
        '''
        Discriminator for generated/real images.

        Parameters
        ----------
        num_filters: int 
            base number of filters used, by default 32
        '''
        super(DiscriminatorImage,self).__init__()

        self.conv_img = nn.Sequential(
            nn.Conv2d(n_channel,n_disc,4,2,1),
        )
        self.conv_l = nn.Sequential(
            nn.ConvTranspose2d(n_l*n_age+n_gender, n_l*n_age+n_gender, 64, 1, 0),
            nn.ReLU()
        )
        self.total_conv = nn.Sequential(
            nn.Conv2d(n_disc+n_l*n_age+n_gender,n_disc*2,4,2,1),
            nn.ReLU(),
            nn.Conv2d(n_disc*2,n_disc*4,4,2,1),
            nn.ReLU(),
            nn.Conv2d(n_disc*4,n_disc*8,4,2,1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8*8*img_size,1024),
            nn.ReLU()
            nn.Linear(1024,1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass of image discriminator.

        Parameters
        ----------
        x: torch.Tensor
            input latent, a tensor of shape (?, latent_dim)
        
        Returns
        -------
        torch.Tensor: real/fake activations, a vector of shape (?, 1)
        '''
       
        conv_img = self.conv_img(img)
        conv_l   = self.conv_l(y)
        catted   = torch.cat((conv_img,conv_l),dim=1)
        total_conv = self.total_conv(catted).view(-1,8*8*img_size)
        out = self.fc(total_conv)
        return out 


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
            nn.ReLU(),
            nn.Linear(num_filters*4,num_filters*2),
            nn.ReLU(),
            nn.Linear(num_filters*2,num_filters),
            nn.ReLU(),
            nn.Linear(num_filters,1),
            nn.Sigmoid()
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
        torch.Tensor: real/fake activations, a vector of shape (?, 1)
        '''
        return self.layers(z)