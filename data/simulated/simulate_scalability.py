from src.simulation import simulate_data
import torch
import numpy as np
import random
import os

n_genes = 50

for x in ['var_molecules/', 'var_factors/', 'var_cells/']:
    if not os.path.isdir(os.path.join('data/scalability/', x)):
        os.makedirs(os.path.join('data/scalability/', x))

# varying number of molecules per cell
for intensity_scale in [50*i for i in range(1, 9)]:
    for i in range(10):
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)

        data = simulate_data(
            n_genes=n_genes,
            n_cells=1,
            n_factors=3,
            factor_images_dir='factors/',
            intensity_scales=torch.tensor([intensity_scale]).repeat(1, n_genes),
            weight_sparsity=0.7,
        )

        torch.save(
            data,
            'data/scalability/var_molecules/{}_{}.pt'.format(intensity_scale, i)
            )

# varying number of factors
for n_factors in [i for i in range(2, 8)]:
    for i in range(10):
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)

        data = simulate_data(
            n_genes=n_genes,
            n_cells=1,
            n_factors=n_factors,
            factor_images_dir='factors/',
            intensity_scales=torch.tensor([100]).repeat(1, n_genes),
            weight_sparsity=0.7,
        )

        torch.save(
            data,
            'data/scalability/var_factors/{}_{}.pt'.format(n_factors, i)
            )

# varying number of cells
for n_cells in [10*i for i in range(1, 6)]:
    for i in range(10):
        torch.manual_seed(i)
        np.random.seed(i)
        random.seed(i)

        data = simulate_data(
            n_genes=n_genes,
            n_cells=n_cells,
            n_factors=3,
            factor_images_dir='factors/',
            intensity_scales=torch.tensor([100]).repeat(n_cells, n_genes),
            weight_sparsity=0.7,
        )

        torch.save(
            data,
            'data/scalability/var_cells/{}_{}.pt'.format(n_cells, i)
            )