import torch
from src import simulation
import os

masks = torch.load('../../data/3t3/masks.pkl')

for intensity_scale in [50, 100, 200, 300]:
    if not os.path.isdir('data/intensity_scale_{}'.format(intensity_scale)):
        os.makedirs('data/intensity_scale_{}/'.format(intensity_scale))

    for cell in range(20):
        torch.manual_seed(cell)

        data = simulation.simulate_data(
            n_genes=50,
            n_factors=3,
            masks=masks[cell].unsqueeze(0),
            intensity_scales=torch.tensor([intensity_scale]).repeat(1, 50),
            factor_lengthscales=[0.5, 0.5, 0.5],
            weight_sparsity=0.7,
            factor_sparsity=1.,
            factor_smoothness=1.5,
        )

        torch.save(
            data,
            'data/intensity_scale_{}/cell_{}.pkl'.format(intensity_scale, cell),
            )