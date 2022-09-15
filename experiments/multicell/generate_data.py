import torch
from src import simulation
import os

masks = torch.load('../../data/3t3/masks.pkl')
n_cells = 20
n_genes = 50

if not os.path.isdir('data/'):
    os.makedirs('data/')

for dataset in range(10):
    dataset_masks = masks[dataset*n_cells:(dataset*n_cells+n_cells)]

    torch.manual_seed(dataset)

    data = simulation.simulate_data(
        n_genes=n_genes,
        n_factors=3,
        masks=dataset_masks,
        intensity_scales=torch.tensor([100]).repeat(n_cells, n_genes),
        factor_lengthscales=torch.tensor([0.5, 0.5, 0.5]).repeat(n_cells, 1, 1, 1),
        weight_sparsity=0.7,
        factor_sparsity=1.,
        factor_smoothness=1.5,
    )

    torch.save(
        data,
        'data/dataset_{}.pkl'.format(dataset),
        )