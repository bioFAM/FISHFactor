from typing import Sequence, Union, Tuple
import torch
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, gaussian_kde
from scipy.optimize import linear_sum_assignment


def gp_percentiles(
    model,
    coords: torch.tensor,
    percentiles: Sequence[float]=[0.5],
    n_samples: int=10,
    ) -> Sequence[torch.tensor]:
    """Get percentiles from the GP posterior.

    :param model: GPyTorch ApproximateGP model
    :param coords: tensor of shape (n_points, 2) with coordinates where the GP
        is evaluated
    :param percentiles: list with percentile values
    :param n_samples: number of samples to use for the estimate
    :returns: List with percentile tensors
    """
    model.eval()
    with torch.no_grad():
        dist = model(coords)
        samples = dist(torch.Size([n_samples]))

    samples = samples.sort(dim=0)[0]
    percentile_samples = [
        samples[int(n_samples * percentile)] for percentile in percentiles
    ]

    return list(percentile_samples)

def spatial_binning(
    data: pd.DataFrame,
    nx: int,
    ny: int,
    xlim: Tuple[float, float]=[0., 1.],
    ylim: Tuple[float, float]=[0., 1.],
    ) -> torch.tensor:
    """Count data points in bins.

    :param data: tensor of shape (N, 3) with x coordinate in first column, y
        coordinate in second column and feature id in third column
    :param nx: number of bins in x direction
    :param ny: number of bins in y direction
    :param xlim: x limits
    :param ylim: y limits
    :returns: tensor of shape (n_features, nx, ny) with bin counts
    """
    data['feature_id'], features = pd.factorize(data['feature'])
    data = data[['x', 'y', 'feature_id']].values

    binned_data = np.zeros([features.shape[0], ny, nx])
    for d in range(features.shape[0]):
        binned_data[d] = np.histogram2d(
            x=data[data[:, 2] == d][:, 1],
            y=data[data[:, 2] == d][:, 0],
            bins=[ny, nx],
            range=[[ylim[0], ylim[1]], [xlim[0], xlim[1]]],
        )[0]

    return torch.tensor(binned_data)

def optimal_assignment(
    x1: torch.tensor,
    x2: torch.tensor,
    item_dim: int,
    ) -> Union[np.array, np.array]:
    """Optimally assign x2 items to x1 by maximizing correlation

    :param x1: first (fixed) tensor
    :param x2: second tensor
    :param item_dim: assignment dimension in x1 and x2 
    :returns: optimal assignment indices of x2 items, correlation coefficients
    """
    correlation = torch.zeros([x1.shape[item_dim], x2.shape[item_dim]])
    for i in range(x1.shape[item_dim]):
        for j in range(x2.shape[item_dim]): 
            correlation[i, j] = pearsonr(
                torch.narrow(x1, item_dim, i, 1).flatten(),
                torch.narrow(x2, item_dim, j, 1).flatten()
                )[0]

    row_ind, col_ind = linear_sum_assignment(-1 * torch.abs(correlation))
    return col_ind, correlation[row_ind, col_ind].numpy()

def normalize_group_coordinates(
    data: torch.tensor,
    ):
    """Rescale the coordinates of every group to the unit square.
    
    :param data: tensor of shape (N, 4) with x and y coordinates in first and
        second column, and enumerated group in fourth column.
    """
    for group_id in data[:, 3].unique():
        group_inds = data[:, 3] == group_id
        x_max = data[group_inds][:, 0].max()
        x_min = data[group_inds][:, 0].min()
        y_max = data[group_inds][:, 1].max()
        y_min = data[group_inds][:, 1].min()

        data[group_inds, 0] = (data[group_inds, 0] - x_min) / (x_max - x_min)
        data[group_inds, 1] = (data[group_inds, 1] - y_min) / (y_max - y_min)

    return data

def grid(
    res: float, 
    xlim: Tuple[float]=(0., 1.),
    ylim: Tuple[float]=(0., 1.)
    ):
    """Generate grid coordinates.
    
    :param res: number of grid points per unit interval per dimension
    :param xlim: x limits
    :param ylim: y limits
    """
    lx = xlim[1] - xlim[0]
    ly = ylim[1] - ylim[0]
    x_axis = torch.linspace(xlim[0], xlim[1], int(res * lx))
    y_axis = torch.linspace(ylim[0], ylim[1], int(res * ly))
    xx, yy = torch.meshgrid([x_axis, y_axis])
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)[:, [1, 0]]
    return grid

def density_mask(
    data: torch.tensor,
    grid: torch.tensor,
    min_density: float=0.1
    ):
    """Generate a foreground mask of the data with a kernel density estimate.

    :param data: tensor of shape (N, 2) with x and y coordinates in first and
        second columns
    :param grid: tensor of shape (n_grid_points, 2) with x and y coordinates of
        grid in first and second columns
    :param min_density: minimum required density of kernel density estimate to
        count grid point as foreground
    """
    res_x = int(grid[:, 0].unique().shape[0]
        / (grid[:, 0].max() - grid[:, 0].min()).item())
    res_y = int(grid[:, 1].unique().shape[0]
        / (grid[:, 1].max() - grid[:, 1].min()).item())

    kernel = gaussian_kde(data[:, [0, 1]].T.cpu().numpy())
    grid_density = kernel(grid[:, [0, 1]].T.cpu().numpy())
    mask = ((grid_density > min_density).T).reshape(res_x, res_y)
    mask = torch.tensor(mask, dtype=torch.bool)
    return mask
