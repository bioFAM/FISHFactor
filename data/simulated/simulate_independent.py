from src.simulation import simulate_data
import torch
import numpy as np
import random
import os

n_genes = 50

if not os.path.isdir('data/independent/'):
    os.makedirs('data/independent/')

for intensity_scale in [50, 100, 200, 300, 400]:
    intensity_scales = torch.tensor([intensity_scale]).repeat(1, n_genes)

    for i in range(20):
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)

        data = simulate_data(
            n_genes=n_genes,
            n_cells=1,
            n_factors=3,
            factor_images_dir='factors/',
            intensity_scales=intensity_scales,
            weight_sparsity=0.7,
        )

        torch.save(
            data,
            'data/independent/{}_{}.pt'.format(intensity_scale, i)
            )