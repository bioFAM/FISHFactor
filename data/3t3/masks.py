import torch
import pandas as pd
from src import utils

grid_res = 50
mask_threshold = 0.5

data = pd.read_feather('../3t3/data_preprocessed.feather')
data = data.rename(columns={'cell' : 'cell_id', 'gene' : 'gene_id'})
data = utils.rescale_cells(data)
grid = utils.grid(grid_res)
masks = utils.cell_masks(data, grid, threshold=mask_threshold)

torch.save(masks, 'masks.pkl')