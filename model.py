import torch
from torch import nn
from torch.nn import functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()
        
        # Build encoder
        self.encoder = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(input_dim, 512),  #784->512
            nn.LeakyReLU (), 
            nn.BatchNorm1d(512),
            nn.Linear(512, 256), # 512->256
            nn.LeakyReLU (), 
            nn.BatchNorm1d(256),
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), # latent_dim->256
            nn.LeakyReLU (),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512), # 256->512
            nn.LeakyReLU (),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim), # 512->784(28x28)
            nn.Sigmoid()
        )


    def encode(self, x):
        # q_phi(z|x)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)


    def decode(self, x):
        return self.decoder(x)


    def parameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.parameterize(mu, logvar)
        return [self.decode(z), mu, logvar]

    
    def compute_loss(self, recons, input, mu, logvar, current_epoch, warmup_epoch=4, beta=0.001):
        # Calculate Reconstruction Loss
        recons_reshaped = recons.view(-1, 1, 28, 28)
        recons_loss = F.mse_loss(recons_reshaped, input)

        # Calculate KL Divergence Loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1).mean()

        total_loss = recons_loss + beta * kld_loss

        return total_loss, recons_loss, kld_loss
