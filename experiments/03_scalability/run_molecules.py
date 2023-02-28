import torch
import sys
from src import fishfactor
import random
import numpy as np

device = sys.argv[1]

intensity_scales = [50*i for i in range(1, 9)]
n_datasets = 10

for intensity_scale in intensity_scales:
    for dataset in range(n_datasets):
        data = torch.load(
            '../../data/simulated/data/scalability/var_molecules/{}_{}.pt'
            .format(intensity_scale, dataset))

        torch.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        model = fishfactor.FISHFactor(
            data=data['coordinates'],
            n_factors=3,
            device=device,
            n_inducing=100,
            grid_res=50,
            factor_smoothness=1.5,
            masks_threshold=0.4,
            init_bin_res=5,
            ).to(device=device)

        model.inference(
            lr=5e-3,
            lrd=1.,
            n_particles=15,
            early_stopping_patience=100,
            min_improvement=0.001,
            max_epochs=10000,
            save=True,
            save_every=100,
            save_dir='results/var_molecules/{}_{}/'
                .format(intensity_scale, dataset),
            )