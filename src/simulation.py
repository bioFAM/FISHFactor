import torch
import numpy as np
import pandas as pd
import random
import string
import os
import imageio.v2 as imageio


def simulate_data(
    n_genes: int,
    n_cells : int,
    n_factors: int,
    factor_images_dir: str,
    intensity_scales: torch.tensor,
    weight_sparsity: float=0.7,
    ):
    """Simulation of single-molecule resolved subcellular gene expression.
    
    Args:
        n_genes: number of genes
        n_cells: number of cells
        n_factors: number of factors per cell
        factor_images_dir: directory with factor *.png files
        intensity_scales: tensor of shape (n_cells, n_genes) with intensity
            scales per cell and gene
        weight_sparsity: Bernoulli probability for weight matrix entries to
            be non-zero

    Returns:
        dictionary with simulated data
    """
    assert intensity_scales.shape[0] == n_cells
    assert intensity_scales.shape[1] == n_genes
    
    # weight sparsity
    while True:
        s = torch.distributions.Bernoulli(
            torch.full([n_genes, n_factors], weight_sparsity)
        ).sample()

        if (s.sum(dim=1) > 0).all(): # >= one factor active for every gene
            break

    # weights
    q = torch.distributions.Normal(
        loc=torch.zeros([n_genes, n_factors]),
        scale=torch.full([n_genes, n_factors], 1.),
    ).sample()

    w = torch.nn.Softplus()(q) * s
    w = w / w.sum(dim=1, keepdim=True) # weights sum to 1 over factors

    # factors (load manually designed factors)
    factor_images = []
    factor_images_files = os.listdir(factor_images_dir)
    for file in factor_images_files:
        image = imageio.imread(os.path.join(factor_images_dir, file))
        image = (image - image.min()) / (image.max() - image.min())
        factor_images.append(image)
    factor_images = np.stack(factor_images, axis=0)

    z = np.zeros([n_cells, n_factors, factor_images.shape[1],
        factor_images.shape[2]])

    for cell in range(n_cells):
        # choose random factors
        factor_inds = np.random.choice(
            a=np.arange(factor_images.shape[0]),
            size=n_factors,
            replace=False
            )
        z[cell] = factor_images[factor_inds]

        # apply random transformations (rotations, flips)
        for factor in range(n_factors):
            z[cell, factor] = np.rot90(
                z[cell, factor],
                k=np.random.randint(0, 4)
                )
            if np.random.randint(0, 1) > 0.5:
                z[cell, factor] = np.flipud(z[cell, factor])
            if np.random.randint(0, 1) > 0.5:
                z[cell, factor] = np.fliplr(z[cell, factor])

    z = torch.tensor(z, dtype=torch.float32)

    # Poisson process intensity
    intensity = torch.matmul(w.unsqueeze(0), z.view(n_cells, n_factors, -1))
    intensity = intensity.view(n_cells, n_genes, z.shape[-2], z.shape[-1])
    intensity *= intensity_scales.view(n_cells, n_genes, 1, 1)

    # generate random gene names
    gene_names = []
    for gene_id in range(n_genes):
        while True:
            name = ''.join(random.choices(
                string.ascii_letters + string.digits, k=4))
            if not name in gene_names:
                gene_names.append(name)
                break
    gene_names.sort()

    # simulation by thinning
    coordinates = []
    for cell in range(n_cells):
        for gene_id in range(n_genes):            
            # homogeneous Poisson process
            max_intensity = intensity[cell, gene_id].max()
            num_coordinates = max(int(torch.poisson(max_intensity).item()), 1)

            coordinates_x = torch.rand([num_coordinates])
            coordinates_y = torch.rand([num_coordinates])

            # thinning
            thinning_probs = (1 - intensity[cell, gene_id] / max_intensity)
            thinning_probs_per_molecule = thinning_probs[
                (coordinates_y * z.shape[-2]).long(),
                (coordinates_x * z.shape[-1]).long()
                ]

            while True:
                rand_probs = torch.rand_like(thinning_probs_per_molecule)
                keep_coordinates = rand_probs >= thinning_probs_per_molecule
                
                if keep_coordinates.sum() >= 1:
                    break

            cell_gene_coordinates = pd.DataFrame({
                'x' : coordinates_x[keep_coordinates],
                'y' : coordinates_y[keep_coordinates],
                'gene' : gene_names[gene_id],
                'cell' : torch.full_like(
                    coordinates_x[keep_coordinates], cell
                    ),
            })
        
            coordinates.append(cell_gene_coordinates)
    coordinates = pd.concat(coordinates, axis=0)

    coordinates.x = coordinates.x.astype('float32')
    coordinates.y = coordinates.y.astype('float32')
    coordinates.gene = coordinates.gene.astype('str')
    coordinates.cell = coordinates.cell.astype('int32')

    return {
        'n_genes' : n_genes,
        'intensity_scales' : intensity_scales,
        'weight_sparsity' : weight_sparsity,
        'n_factors' : n_factors,
        'n_cells' : n_cells,
        's' : s,
        'q' : q,
        'w' : w,
        'z' : z,
        'intensity' : intensity,
        'coordinates' : coordinates,
        'gene_names' : gene_names,
        }