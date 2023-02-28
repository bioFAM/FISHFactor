import torch
from src import fishfactor
import random
import numpy as np
import pandas as pd
import sys

n_cells = 25

device = sys.argv[1]
data = pd.read_feather('../../data/application/data_preprocessed.feather')

# select genes with an average of more than 30 counts per cell
gene_count = (data.groupby(['gene']).size().to_frame('count')
    .sort_values(by='count').reset_index())
filtered_genes = (gene_count.query('count > ({} * 30)'
    .format(data.cell.unique().shape[0])).gene.values)
data = data[data.gene.isin(filtered_genes)]
data = data[data.cell<n_cells]

torch.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

model = fishfactor.FISHFactor(
    data=data,
    n_factors=3,
    n_inducing=100,
    grid_res=50,
    factor_smoothness=1.5,
    masks_threshold=0.4,
    init_bin_res=5,
    device=device,
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
    save_dir='results/full/'
)