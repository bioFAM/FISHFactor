import torch
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, pearsonr
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF
from typing import Tuple, Union

def rescale_cells(
    data: pd.DataFrame,
    ) -> pd.DataFrame:
    """Rescale the coordinates of every cell to the unit square
    
    Args:
        data: DataFrame with columns 'x', 'y', 'cell_id'
    
    Returns:
        DataFrame with rescaled 'x' and 'y' columns
    """
    n_cells = data.cell_id.unique().shape[0]
    rescaled_cell_list = []

    for cell_id in range(n_cells):
        cell_data = data[data.cell_id==cell_id]
        x_range = cell_data.x.max() - cell_data.x.min()
        y_range = cell_data.y.max() - cell_data.y.min()
        scale_factor = max(x_range, y_range)
        x_rescaled = (cell_data.x - cell_data.x.min()) / scale_factor
        y_rescaled = (cell_data.y - cell_data.y.min()) / scale_factor
        rescaled_cell_data = cell_data.copy()
        rescaled_cell_data.x = x_rescaled
        rescaled_cell_data.y = y_rescaled
        rescaled_cell_list.append(rescaled_cell_data)

    return pd.concat(rescaled_cell_list, axis=0)

def grid(
    resolution: float,
    ) -> torch.tensor:
    """Generate grid coordinates
    
    Args:
        resolution: number of grid points per dimension
    
    Returns:
        tensor of shape (resolution**2, 2) with x and y coordinates
    """

    axis = torch.linspace(0, 1, resolution)
    xx, yy = torch.meshgrid([axis, axis], indexing="ij")
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    grid = grid.to(dtype=torch.float32)

    return grid[:, [1, 0]]

def cell_masks(
    data: pd.DataFrame,
    grid: torch.tensor,
    threshold: float=0.1,
    ) -> torch.tensor:
    """Generate a cell mask for every cell based on molecule density.
    
    Args:
        data: DataFrame with columns 'x', 'y', 'cell_id'
        grid: grid coordinates with shape (resolution**2, 2)
        threshold: KDE threshold for mask
        
    Returns:
        tensor of shape (n_cells, nx, ny) with binary cell masks
    """
    masks = []
    n_cells = data.cell_id.unique().shape[0]
    res = int(np.sqrt(grid.shape[0]))

    for cell_id in range(n_cells):
        cell_data = data[data.cell_id==cell_id]
        kernel = gaussian_kde(
            np.stack([cell_data.x.values, cell_data.y.values], 0)
            )
        grid_density = kernel(grid.cpu().numpy().T)
        mask = (grid_density > threshold).reshape(res, res)
        mask = torch.tensor(mask, dtype=torch.bool)
        masks.append(mask)
    
    return torch.stack(masks, dim=0)

def initialize_weights(
    data: pd.DataFrame,
    n_factors: int,
    bin_res: int=5,
    ) -> torch.tensor:
    """Initialize the weight matrix using NMF on binned data.
    
    Args:
        data: must contain 'x', 'y', 'gene id', 'cell_id' columns
        n_factors: number of latent factors
        bin_res: number of bins per spatial dimension

    Returns:
        weight matrix, tensor of shape (n_genes, n_factors)
    """
    xlim = [data.x.min(), data.x.max()]
    ylim = [data.y.min(), data.y.max()]
    n_cells = data.cell_id.unique().shape[0]
    n_genes = data.gene_id.unique().shape[0]
    binned_data = []

    for cell_id in range(n_cells):
        binned_cell_data = []
        for gene_id in range(n_genes):
            cell_gene_data = data[(data.gene_id==gene_id) & (data.cell_id==cell_id)]
            histogram, _, _ = np.histogram2d(
                x=cell_gene_data.x,
                y=cell_gene_data.y,
                bins=[bin_res, bin_res],
                range=[xlim, ylim],
                )
            binned_cell_data.append(histogram.flatten())
        binned_data.append(np.stack(binned_cell_data, axis=0))
    binned_data = np.stack(binned_data, axis=1)

    nmf = NMF(n_components=n_factors, init='nndsvd', max_iter=100000)
    nmf.fit(binned_data.reshape(binned_data.shape[0], -1).T)
    weights = torch.tensor(nmf.components_.T, dtype=torch.float32)

    weights /= weights.max(dim=0)[0].view(1, -1)

    return weights

def average_intensity(
    data: pd.DataFrame,
    masks: torch.tensor,
    per_gene: True,
    per_cell: True,
    ) -> torch.tensor:
    """Compute the average intensity within the cell masks
    
    Args:
        data: DataFrame with 'gene_id' and 'cell_id' columns
        masks: binary cell masks with shape (n_cells, nx, ny)
        per_gene: whether the intensity is computed for genes individually
        per_cell: whether the intensity is computed for cells individually
        
    Returns:
        average intensity with shape (n_genes, n_cells)
    """
    area = masks.sum(dim=[-1, -2]) / (masks.shape[1] * masks.shape[2])
    n_genes = data.gene_id.unique().shape[0]
    n_cells = data.cell_id.unique().shape[0]

    if per_gene:
        if per_cell:
            n_molecules = data.groupby(['gene_id', 'cell_id']).size().reset_index()
            table = n_molecules.pivot('gene_id', 'cell_id', 0).fillna(0).values
            avg_intensity = torch.tensor(table) / area.view(1, n_cells)
        else:
            n_molecules = data.groupby(['gene_id']).size() / n_cells
            avg_intensity = torch.tensor(n_molecules.values).view(n_genes, 1) / area.view(1, n_cells)
            
    else:
        if per_cell:
            n_molecules = data.groupby(['cell_id']).size() / n_genes
            avg_intensity = torch.tensor(n_molecules.values).view(1, n_cells) / area.view(1, n_cells)
            avg_intensity = avg_intensity.repeat([n_genes, 1])

        else:
            n_molecules = data.shape[0] / (n_genes * n_cells)
            avg_intensity = torch.tensor([n_molecules]).view(1, 1) / area.view(1, n_cells)
            avg_intensity = avg_intensity.repeat([n_genes, 1])

    return avg_intensity.to(dtype=torch.float32)

def optimal_assignment(
    x1: torch.tensor,
    x2: torch.tensor,
    assignment_dim: int,
    ) -> Union[np.array, np.array]:
    """Find the permutation of x2 elements along the selected dimension to
        maximize correlation with x1.

    Args:
        x1: first (fixed) tensor
        x2: second tensor
        assignment_dim: assignment dimension
    
    Returns:
        optimal assignment indices of x2 items, correlation coefficients
    """
    correlation = torch.zeros([x1.shape[assignment_dim], x2.shape[assignment_dim]])
    for i in range(x1.shape[assignment_dim]):
        for j in range(x2.shape[assignment_dim]): 
            correlation[i, j] = pearsonr(
                torch.narrow(x1, assignment_dim, i, 1).flatten(),
                torch.narrow(x2, assignment_dim, j, 1).flatten()
                )[0]
    correlation = torch.nan_to_num(correlation, 0)

    row_ind, col_ind = linear_sum_assignment(-1 * torch.abs(correlation))
    return col_ind, correlation[row_ind, col_ind].numpy()

def binning(
    x: torch.tensor,
    y: torch.tensor,
    resolution: float,
    xlim: Tuple[float, float]=[0., 1.],
    ylim: Tuple[float, float]=[0., 1.],
    ) -> torch.tensor:
    """Aggregate coordinates in spatial bins
    
    Args:
        x: x coordinates
        y: y coordinates
        resolution: number of bins per unit interval in every dimension
        xlim: limits of x coordinate
        ylim: limits of y coordinate
        
    Returns:
        tensor of shape (nx, ny) with bin counts
    """
    nx = int((xlim[1] - xlim[0]) * resolution)
    ny = int((ylim[1] - ylim[0]) * resolution)

    binned_data = np.histogram2d(
        x=y.numpy(),
        y=x.numpy(),
        bins=[ny, nx],
        range=[ylim, xlim],
    )[0]

    binned_data = torch.tensor(
        data=binned_data,
        dtype=torch.int32,
    )

    return binned_data