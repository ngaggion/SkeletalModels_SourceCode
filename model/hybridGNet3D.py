import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import ChebConv, Pool, residualBlock3D, scipy_to_torch_sparse
import numpy as np
import pickle as pkl
import scipy.sparse as sp

class Encoder(nn.Module):
    def __init__(self, latents = 64, h = 256, w = 256, slices = 256):
        super(Encoder, self).__init__()
               
        self.residual1 = residualBlock3D(in_channels=1, out_channels=8)
        self.residual2 = residualBlock3D(in_channels=8, out_channels=16)
        self.residual3 = residualBlock3D(in_channels=16, out_channels=32)
        self.residual4 = residualBlock3D(in_channels=32, out_channels=64)
        self.residual5 = residualBlock3D(in_channels=64, out_channels=128)
        self.residual6 = residualBlock3D(in_channels=128, out_channels=128)

        # Input shape is slices x h x w
        self.maxpool = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

        h2 = h  // 32
        w2 = w  // 32
        s2 = slices  // 32

        self.mu = nn.Linear(128*h2*w2*s2, latents)
        self.sigma = nn.Linear(128*h2*w2*s2, latents)

    def forward(self, x):

        x = self.residual1(x)
        x = self.maxpool(x)
        
        x = self.residual2(x)
        x = self.maxpool(x)
        
        x = self.residual3(x)
        x = self.maxpool(x)
        
        x = self.residual4(x)
        x = self.maxpool(x)
        
        x = self.residual5(x)
        x = self.maxpool(x)
        
        x = self.residual6(x)

        x = x.view(x.size(0), -1)
        
        mu = self.mu(x)
        sigma = self.sigma(x)

        return mu, sigma

class GConv(nn.Module):
    def __init__(self, in_channels, out_channels, K = 6):
        super(GConv, self).__init__()

        self.gConv= ChebConv(in_channels, out_channels, K)
        self.norm = torch.nn.InstanceNorm1d(out_channels)

    def forward(self, x, adj_indices):
        x = self.gConv(x, adj_indices)
        x = self.norm(x)
        x = F.relu(x)

        return x

class HybridGNet3D(nn.Module):
    def __init__(self, config):
        super(HybridGNet3D, self).__init__()
        
        self.device = config['device']
        
        # Some encoder hyperparameters
        config['h'] = 256
        config['w'] = 256
        config['slices'] = 256
        self.latents = 64
        
        self.encoder = Encoder(latents = self.latents, h = config['h'], w = config['w'], slices = config['slices'])
        
        # Load adjacency matrices
        A = sp.load_npz("adjacency.npz")
        # Convert scipy sparse matrices to PyTorch sparse tensors, move them to the device and convert to float
        self.adjacency_matrices = [scipy_to_torch_sparse(A).to(self.device).float()]
        n_nodes = [A.shape[0]]

        # Filters of Graph convolutional layers
        self.filters = [3, 16, 32, 32, 64, 64]
        
        self.K = 6
        
        # Fully connected decoder
        self.linear_decoder = torch.nn.Linear(self.latents, self.filters[5] * n_nodes[-1])
        torch.nn.init.normal_(self.linear_decoder.weight, 0, 0.1) 

        # Graph convolutional decoder
        self.unpool = Pool()

        self.GC1 = GConv(self.filters[5], self.filters[4], self.K)
        self.GC2 = GConv(self.filters[4], self.filters[3], self.K)
        self.GC3 = GConv(self.filters[3], self.filters[2], self.K)
        self.GC4 = GConv(self.filters[2], self.filters[1], self.K)
        self.GC5 = GConv(self.filters[1], self.filters[1], self.K)
        self.GCout = ChebConv(self.filters[1], self.filters[0], self.K, bias = False) 
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) 

    def forward(self, x):
        self.mu, self.log_var = self.encoder(x)
        
        if self.training:
            z = self.sampling(self.mu, self.log_var)
        else:
            z = self.mu

        x = self.linear_decoder(z)
        x = F.relu(x)
        
        x = x.reshape(x.shape[0], -1, self.filters[5])
                
        x = self.GC1(x, self.adjacency_matrices[0]._indices())
        x = self.GC2(x, self.adjacency_matrices[0]._indices())
        x = self.GC3(x, self.adjacency_matrices[0]._indices())
        x = self.GC4(x, self.adjacency_matrices[0]._indices())
        x = self.GC5(x, self.adjacency_matrices[0]._indices()) 
        x = self.GCout(x, self.adjacency_matrices[0]._indices()) # Sin relu y sin bias
        
        return x