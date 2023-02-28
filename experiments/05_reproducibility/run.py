import torch
import sys
import random
import numpy as np
from src import fishfactor

device = sys.argv[1]

for intensity_scale in [50, 100, 200, 300, 400]:
    data = torch.load(
        '../../data/simulated/data/shared/0_{}.pt'
        .format(intensity_scale))

    coordinates = data['coordinates']

    for i in range(10):
        torch.manual_seed(i)
        random.seed(i)
        np.random.seed(i)

        model = fishfactor.FISHFactor(
            data=coordinates,
            n_factors=3,
            device=device,
            n_inducing=100,
            grid_res=50,
            factor_smoothness=1.5,
            masks_threshold=0.4,
            init_bin_res=5
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
            save_dir='results/{}_{}/'.format(intensity_scale, i),
        )