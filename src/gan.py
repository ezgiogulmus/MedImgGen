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


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dims, image_dim, norm=True, activation="leaky_relu"):
        super(Generator, self).__init__()
        self.image_dim = image_dim
        self.net = create_network(latent_dim, hidden_dims, int(np.prod(self.image_dim)), norm, activation)
        self.net.add_module("final_activation", nn.Tanh())

    def forward(self, x):
        x = self.net(x)
        return x.view(-1, 1, *self.image_dim)


class Discriminator(nn.Module):
    def __init__(self, image_dim, hidden_dims, activation="leaky_relu"):
        super(Discriminator, self).__init__()
        self.image_dim = image_dim
        self.net = create_network(int(np.prod(self.image_dim)), hidden_dims, 1, norm=False, activation=activation)
        self.net.add_module("final_activation", nn.Sigmoid())

    def forward(self, x):
        x = x.view(-1, int(np.prod(self.image_dim)))
        return self.net(x)


class GAN_Trainer:
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

        self.bce_loss = nn.BCELoss()
        self.generator = Generator(self.latent_dim, hidden_dims, self.image_dim, norm=True if self.batch_size > 1 else False, activation=activation)
        self.discriminator = Discriminator(self.image_dim, hidden_dims[::-1], activation=activation)

        self.generator.to(device)
        self.discriminator.to(device)
        self.bce_loss.to(device)

        self.gen_opt = optim.Adam(self.generator.parameters(), lr=lr, weight_decay=l2_reg)
        self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=l2_reg)

        self.fid_metric = FrechetInceptionDistance(device=device) if calculate_fid else None

    def train_step(self, nb_epochs=50, print_batch=100, save_batch=500, run_name=""):
        for e in range(nb_epochs):
            start = time.time()
            print(f"Epoch: {e}/{nb_epochs}")
            for i, (img_batch, _) in enumerate(self.dataloader):
                img_batch = img_batch.to(device)

                # Train Generator
                self.gen_opt.zero_grad()
                z = torch.randn(img_batch.size(0), self.latent_dim).to(device)
                fake_batch = self.generator(z)
                fake_preds = self.discriminator(fake_batch)

                gen_loss = self.bce_loss(fake_preds, torch.ones_like(fake_preds))
                gen_loss.backward()
                self.gen_opt.step()
                gen_loss = gen_loss.item()

                # Train Discriminator
                self.disc_opt.zero_grad()
                with torch.no_grad():
                    fake_batch = self.generator(z).detach()
                real_preds = self.discriminator(img_batch)
                fake_preds = self.discriminator(fake_batch)

                real_loss = self.bce_loss(real_preds, torch.ones_like(real_preds))
                fake_loss = self.bce_loss(fake_preds, torch.zeros_like(fake_preds))
                disc_loss = (real_loss + fake_loss) / 2
                disc_loss.backward()
                self.disc_opt.step()
                disc_loss = disc_loss.item()

                if i % print_batch == 0:
                    print("\tBatch %d/%d | Gen Loss: %.2f | Disc Loss: %.2f" %(i, len(self.dataloader), gen_loss, disc_loss))
                    if self.writer:
                        self.writer.add_scalar('gen_loss', gen_loss, e*len(self.dataloader)+i)
                        self.writer.add_scalar('disc_loss', disc_loss, e*len(self.dataloader)+i)
                if i % save_batch == 0 and i > 0:
                    fid = self.save(f"{run_name}epoch{e}_batch{i}")
                    if not np.isnan(fid):
                        print(f'FID: {fid}')
                    if self.writer:
                        self.writer.add_scalar('fid', fid, e*len(self.dataloader)+i)

            print('Time taken: %.2f sec' %(time.time() - start))
            print()

    def save(self, name):
        print('Saving model checkpoints...')
        torch.save(self.generator.state_dict(), os.path.join(self.results_dir, f"{name}_gen_model.pth"))
        torch.save(self.discriminator.state_dict(), os.path.join(self.results_dir, f"{name}_disc_model.pth"))
        fid = self.save_sample(name)
        return fid

    def save_sample(self, name):
        self.generator.eval()  # Set to evaluation mode
        with torch.no_grad():
            sample_fakes = self.generator(torch.randn(100, self.latent_dim).to(device))
        sample_fakes = sample_fakes.cpu().detach()

        if self.fid_metric is None:
            fid = np.nan
        else:
            real_images = next(iter(self.dataloader))
            real_images = real_images.to(device)
            self.fid_metric.update(to_inception_input(real_images), is_real=True)
            self.fid_metric.update(to_inception_input(sample_fakes), is_real=False)
            fid = self.fid_metric.compute().cpu().numpy()

        print("Saving generated images...")
        for i in range(20):
            plt.imsave(os.path.join(self.save_dir, f"{name}_fake{i}.png"), to_uint8(sample_fakes)[i, 0].numpy(), cmap="gray")

        self.generator.train()
        return fid

    def load_model(self, name):
        print("Loading model checkpoints...")
        self.generator.load_state_dict(torch.load(os.path.join(self.results_dir, f"{name}_gen_model.pth")))
        self.discriminator.load_state_dict(torch.load(os.path.join(self.results_dir, f"{name}_disc_model.pth")))
        print("All checkpoints are matched.")