import torch
import sys
from src import fishfactor
import random
import numpy as np
import os

device = sys.argv[1]
n_cells_list = [2, 3, 4, 5, 10, 15, 20]

for dataset in range(10):
    data = torch.load('data/dataset_{}.pkl'.format(dataset))

    for n_cells in n_cells_list:
        cells_data = data['coordinates'].query('cell < {}'.format(n_cells)).copy()

        if os.path.isfile('results/multicell_models/dataset_{}/{}_cells/final/latents.pkl'
            .format(dataset, n_cells)):
            print('Dataset {} with {} cells already exists, skipping.'
                .format(dataset, n_cells))
            continue

        torch.manual_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        try:
            model = fishfactor.FISHFactor(
                data=cells_data,
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
                save_dir='results/multicell_models/dataset_{}/{}_cells/'
                    .format(dataset, n_cells),
                )

        except:
            continue