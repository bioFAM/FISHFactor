from src.simulation import simulate_data
import torch
import numpy as np
import random
import os

n_genes = 100
n_cells = 1000
intensity_scale = 100

if not os.path.isdir('data/'):
    os.makedirs('data/')

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)

data = simulate_data(
    n_genes=n_genes,
    n_cells=n_cells,
    n_factors=3,
    factor_images_dir='factors/',
    intensity_scales=torch.tensor([intensity_scale]).repeat(n_cells, n_genes),
    weight_sparsity=0.7,
)

torch.save(
    data,
    'data/large.pt'
    )