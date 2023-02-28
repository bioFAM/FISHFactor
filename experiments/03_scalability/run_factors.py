import torch
import sys
from src import fishfactor
import random
import numpy as np

device = sys.argv[1]

n_factors_list = [i for i in range(2, 8)]
n_datasets = 10

for n_factors in n_factors_list:
    for dataset in range(n_datasets):
        data = torch.load(
            '../../data/simulated/data/scalability/var_factors/{}_{}.pt'
            .format(n_factors, dataset))

        torch.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        model = fishfactor.FISHFactor(
            data=data['coordinates'],
            n_factors=n_factors,
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
            save_dir='results/var_factors/{}_{}/'
                .format(n_factors, dataset),
            )