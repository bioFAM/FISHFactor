from src.simulation import simulate_data
import torch
import numpy as np
import random
import os

n_genes = 50
n_cells = 20

if not os.path.isdir('data/shared/'):
    os.makedirs('data/shared/')

for i in range(10):
    for intensity_scale in [50, 100, 200, 300, 400]:
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)

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
            'data/shared/{}_{}.pt'.format(i, intensity_scale)
            )