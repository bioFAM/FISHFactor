import torch
import sys
import random
import numpy as np
from src import fishfactor

device = sys.argv[1]
data = torch.load('../../data/simulated/data/large.pt')

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
    save_dir='results/',
)