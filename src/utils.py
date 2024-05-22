import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch
from torchvision import transforms


# Data utils #############################################
class CXR_Dataset(Dataset):
    def __init__(self, df, image_dim, data_dir=None):
        self.data_dir = data_dir
        self.df = df
        self.transform = transforms.Compose([
            transforms.Resize(image_dim),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.df["paths"].iloc[idx])
        label = self.df["labels"].iloc[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        return image
    
# Model utils #############################################
def create_network(input_dim, hidden_dims, output_dim, norm, activation):
    activation_functions = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU()
    }

    act = activation_functions.get(activation, nn.LeakyReLU())

    layers = [nn.Linear(input_dim, hidden_dims[0])]
    if norm:
        layers.append(nn.BatchNorm1d(hidden_dims[0]))

    for i in range(len(hidden_dims) - 1):
        layers.append(act)
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
        if norm:
            layers.append(nn.BatchNorm1d(hidden_dims[i + 1]))

    layers.append(act)
    layers.append(nn.Linear(hidden_dims[-1], output_dim))

    return nn.Sequential(*layers)

# Utility functions #############################################
def to_uint8(tensor):
    # Convert from [-1, 1] to [0, 255]
    return ((tensor + 1) * 127.5).to(torch.uint8)

def to_inception_input(tensor):
    # Convert to input shape of InceptionV3
    tensor = transforms.Resize((229, 229))(tensor.repeat(1, 3, 1, 1))

    # Convert from [-1, 1] to [0, 1]
    return (tensor + 1) / 2