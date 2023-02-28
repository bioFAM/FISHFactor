import torch
from sklearn.decomposition import NMF
from src import utils
import random
import numpy as np
import os

for intensity_scale in [50, 100, 200, 300, 400]:
    for cell in range(20):
        data = torch.load(
            '../../data/simulated/data/independent/{}_{}.pt'
            .format(intensity_scale, cell))
        coords = data['coordinates'].sort_values(['cell', 'gene'])
        gene_names = coords.gene.unique()

        for resolution in [5, 10, 20, 30, 40]:
            binned_cell_data = torch.zeros(
                [data['n_genes'], resolution, resolution])

            for i, gene in enumerate(gene_names):
                gene_data = coords.query("gene=='{}'".format(gene))

                binned_cell_data[i] = utils.binning(
                    torch.tensor(gene_data.x.values),
                    torch.tensor(gene_data.y.values),
                    xlim=[0, 1],
                    ylim=[0, 1],
                    resolution=resolution,
                    )

            torch.manual_seed(1234)
            random.seed(1234)
            np.random.seed(1234)

            nmf = NMF(n_components=3, init='nndsvd', max_iter=50000)

            z = torch.tensor(
                nmf.fit_transform(
                    binned_cell_data.flatten(start_dim=1).T).T
                    .reshape(1, -1, resolution, resolution))

            z_original = z.clone()
            z = torch.nn.functional.interpolate(z, [50, 50]).squeeze(0)

            w = nmf.components_.T

            results = {
                'z' : z,
                'w' : w,
                'z_original' : z_original,
                'binned_cell_data' : binned_cell_data,
                'intensity_scale' : intensity_scale,
                'resolution' : resolution,
                'cell' : cell,
                }

            os.makedirs('results/nmf/{}_{}/'
                .format(intensity_scale, cell), exist_ok=True)

            torch.save(
                results,
                'results/nmf/{}_{}/{}.pkl'
                    .format(intensity_scale, cell, resolution)
                )