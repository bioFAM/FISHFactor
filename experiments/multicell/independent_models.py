import torch
import sys
from src import fishfactor
import random
import numpy as np

device = sys.argv[1]

for dataset in range(10):
    for cell in range(20):
        data = torch.load('data/dataset_{}.pkl'.format(dataset))
        cell_data = data['coordinates'][data['coordinates'].cell==cell].copy()

        torch.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        try:
            model = fishfactor.FISHFactor(
                data=cell_data,
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
                min_improvement=0.005,
                max_epochs=10000,
                save=True,
                save_every=100,
                save_dir='results/independent_models/dataset_{}/cell_{}/'
                    .format(dataset, cell),
                )

        except:
            continue