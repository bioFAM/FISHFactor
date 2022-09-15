import torch
import pandas as pd
import gpytorch
from src import utils


def simulate_data(
    n_genes: int,
    n_factors: int,
    masks: torch.tensor,
    intensity_scales: torch.tensor,
    factor_lengthscales: torch.tensor,
    weight_sparsity: float=0.7,
    factor_sparsity: float=1.,
    factor_smoothness: float=1.5,
    ) -> dict:
    """Simulation of single-molecule resolved subcellular gene expression.
    
    Args:
        n_genes: number of genes
        n_factors: number of latent factors
        masks: binary cell masks, shape (n_cells, res, res)
        intensity_scales: intensity scale factor, shape (n_cells, n_genes)
        factor_lengthscales: lengthscales of factor GP prior kernel, shape
            (n_cells, n_factors)
        weight_sparsity: probability for weight matrix entries to be non-zero
        factor_sparsity: probability for factors to be non-zero
        factor_smoothness: smoothness parameter of matern kernel (0.5, 1.5, 2.5)

    Returns:
        dictionary with simulated data
    """
    assert intensity_scales.shape[0] == masks.shape[0]
    assert intensity_scales.shape[1] == n_genes

    n_cells = masks.shape[0]

    # weights
    while True:
        s_weights = torch.distributions.Bernoulli(
            torch.full([n_genes, n_factors], weight_sparsity)
        ).sample().unsqueeze(0).repeat(n_cells, 1, 1)
        if (s_weights.sum(dim=2) > 0).all(): # >= one factor active in every gene
            break

    q = torch.distributions.Normal(
        loc=torch.zeros([n_genes, n_factors]),
        scale=torch.full([n_genes, n_factors], 1.),
    ).sample().unsqueeze(0).repeat(n_cells, 1, 1)

    w = torch.nn.Softplus()(q)
    w *= s_weights
    w = w / w.sum(dim=2, keepdim=True)

    # factors
    res = masks.shape[1]
    grid = utils.grid(res)
    
    mean = gpytorch.means.ZeroMean(batch_shape=[n_cells, n_factors])
    kernel = gpytorch.kernels.MaternKernel(
        nu=factor_smoothness,
        batch_shape=[n_cells, n_factors]
        )
    kernel.lengthscale = factor_lengthscales
    dist = gpytorch.distributions.MultivariateNormal(mean(grid), kernel(grid))

    f = dist.sample().view(n_cells, n_factors, res, res)
    z = torch.nn.Softplus()(f)
    z *= masks.unsqueeze(1)
    z /= z.flatten(start_dim=2).max(dim=2)[0].view(n_cells, n_factors, 1, 1)

    s_factors = torch.distributions.Bernoulli(
        torch.full([n_cells, n_factors], factor_sparsity)
        ).sample().view(n_cells, n_factors, 1, 1)
    z *= s_factors

    # Poisson process intensity
    intensity = torch.matmul(w, z.view(n_cells, n_factors, -1))
    intensity *= intensity_scales.view(n_cells, n_genes, 1)
    intensity = intensity.view(n_cells, n_genes, res, res)

    # simulation by thinning
    coordinates = []
    for cell in range(n_cells):
        for gene in range(n_genes):
            # homogeneous Poisson process
            max_intensity = intensity[cell][gene].max()
            num_coordinates = max(int(torch.poisson(max_intensity).item()), 1)

            coordinates_x = torch.rand([num_coordinates])
            coordinates_y = torch.rand([num_coordinates])

            # thinning
            thinning_probs = (1 - intensity[cell][gene] / max_intensity)
            thinning_probs_per_molecule = thinning_probs[
                (coordinates_y * res).long(),
                (coordinates_x * res).long()
                ]
            rand_probs = torch.rand_like(thinning_probs_per_molecule)
            keep_coordinates = rand_probs >= thinning_probs_per_molecule

            if keep_coordinates.sum() < 1:
                ind = torch.randint(high=keep_coordinates.shape[0], size=[1])
                keep_coordinates[ind] = True

            cell_gene_coordinates = pd.DataFrame({
                'x' : coordinates_x[keep_coordinates],
                'y' : coordinates_y[keep_coordinates],
                'gene' : torch.full_like(coordinates_x[keep_coordinates], gene),
                'cell' : torch.full_like(coordinates_x[keep_coordinates], cell),
            })
        
            coordinates.append(cell_gene_coordinates)
    coordinates = pd.concat(coordinates, axis=0)

    return {
        'n_genes' : n_genes,
        'n_factors' : n_factors,
        'n_cells' : n_cells,
        'masks' : masks,
        'intensity_scales' : intensity_scales,
        'factor_lengthscales' : factor_lengthscales,
        'weight_sparsity' : weight_sparsity,
        'factor_sparsity' : factor_sparsity,
        'factor_smoothness' : factor_smoothness,
        's_weights' : s_weights,
        's_factors' : s_factors,
        'q' : q,
        'w' : w,
        'res' : res,
        'grid' : grid,
        'f' : f,
        'z' : z,
        'intensity' : intensity,
        'coordinates' : coordinates,
        }