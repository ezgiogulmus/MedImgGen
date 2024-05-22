import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torcheval.metrics import FrechetInceptionDistance
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from src.utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, norm, activation):
        super(Encoder, self).__init__()
        self.net = create_network(input_dim, hidden_dims, latent_dim*2, norm=norm, activation=activation)
    
    def forward(self, x):
        h = self.net(x)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dims, output_dim, norm, activation):
        super(Decoder, self).__init__()
        self.net = create_network(latent_dim, hidden_dims[::-1], output_dim, norm=norm, activation=activation)
    
    def forward(self, z):
        return self.net(z)


class VAE(nn.Module):
    def __init__(self, image_dim, hidden_dims, latent_dim, norm, activation):
        super(VAE, self).__init__()
        self.image_dim = image_dim

        self.encoder = Encoder(int(np.prod(image_dim)), hidden_dims, latent_dim, norm=norm, activation=activation)
        self.decoder = Decoder(latent_dim, hidden_dims, int(np.prod(image_dim)), norm=norm, activation=activation)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        x = x.view(-1, int(np.prod(self.image_dim)))
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        x_recon = x_recon.view(-1, *self.image_dim)
        return x_recon, mu, logvar

    def loss_function(self, x, x_recon, mu, logvar):
        recon_loss = nn.functional.mse_loss(x_recon, x, reduction='sum')
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kld_loss, recon_loss, kld_loss


class VAE_Trainer:
    def __init__(self, df,
        image_dim, latent_dim, hidden_dims, activation,
        lr, l2_reg, batch_size, data_dir="data/images",
        results_dir="results", log=True,
        calculate_fid=True
        ):

        self.base_dir = os.getcwd()
        self.data_dir = os.path.join(self.base_dir, data_dir)
        self.results_dir = os.path.join(self.base_dir, results_dir)
        self.save_dir = os.path.join(self.results_dir, "generated_images")
        os.makedirs(self.save_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.results_dir) if log else None

        self.batch_size = batch_size
        self.image_dim = image_dim
        self.latent_dim = latent_dim

        dataset = CXR_Dataset(df, self.image_dim, self.data_dir)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        print("Number of batches: ", len(self.dataloader))

        self.model = VAE(self.image_dim, hidden_dims, self.latent_dim, norm=False, activation=activation)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2_reg)
        self.fid_metric = FrechetInceptionDistance(device=device) if calculate_fid else None

    def train_step(self, nb_epochs=50, print_batch=100, save_batch=500, run_name=""):
        for e in range(nb_epochs):
            start = time.time()
            train_loss = 0.0
            for i, (img_batch, _) in enumerate(self.dataloader):
                img_batch = img_batch.to(device)
            
                recon_batch, mu, logvar = self.model(img_batch)
                loss, recon_loss, kld_loss = self.model.loss_function(img_batch, recon_batch, mu, logvar)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                if i % print_batch == 0:
                    print("\tBatch %d/%d | Loss: %.2f | Recon Loss: %.2f | KLD Loss: %.2f" %(i, len(self.dataloader), loss.item(), recon_loss.item(), kld_loss.item()))
                    if self.writer:
                        self.writer.add_scalar('loss', loss.item(), e*len(self.dataloader)+i)
                        self.writer.add_scalar('recon_loss', recon_loss.item(), e*len(self.dataloader)+i)
                        self.writer.add_scalar('kld_loss', kld_loss.item(), e*len(self.dataloader)+i)
                if i % save_batch == 0 and i > 0:
                    fid = self.save(f"{run_name}epoch{e}_batch{i}")
                    if not np.isnan(fid):
                        print(f'FID: {fid}')
                    if self.writer:
                        self.writer.add_scalar('fid', fid, e*len(self.dataloader)+i)

            print(f'====> Epoch: {e}/{nb_epochs} | Average loss: {train_loss / len(self.dataloader.dataset):.4f} | Time taken: {time.time() - start:.2f} sec')
            print()

    def save(self, name):
        print('Saving model checkpoints...')
        torch.save(self.model.state_dict(), os.path.join(self.results_dir, f"{name}_ckpts.pth"))
        fid = self.save_sample(name)
        return fid
    
    def save_sample(self, name):
        print("Saving generated images...")
        self.model.eval()  # Set to evaluation mode
        with torch.no_grad():
            sample_z = torch.randn(100, self.latent_dim).to(device)
            sample_fakes = self.model.decoder(sample_z).view(-1, *self.image_dim)
        sample_fakes = sample_fakes.cpu().detach()
        
        if self.fid_metric is None:
            fid = np.nan
        else:
            real_images, _ = next(iter(self.dataloader))
            real_images = real_images.to(device)
            self.fid_metric.update(to_inception_input(real_images), is_real=True)
            self.fid_metric.update(to_inception_input(sample_fakes), is_real=False)
            fid = self.fid_metric.compute().cpu().numpy()

        print("Saving generated images...")
        for i in range(20):
            plt.imsave(os.path.join(self.save_dir, f"{name}_fake{i}.png"), to_uint8(sample_fakes)[i, 0].numpy(), cmap="gray")

        self.model.train()
        return fid
        
    def load_model(self, name):
        print("Loading model checkpoints...")
        self.model.load_state_dict(torch.load(os.path.join(self.results_dir, f"{name}_ckpts.pth")))
        print("All checkpoints are matched.")