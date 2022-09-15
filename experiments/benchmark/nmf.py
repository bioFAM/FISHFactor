import torch
from sklearn.decomposition import NMF
from src import utils
import random
import numpy as np
import os

for intensity_scale in [50, 100, 200, 300]:
    for cell in range(20):
        data = torch.load('data/intensity_scale_{}/cell_{}.pkl'
            .format(intensity_scale, cell))

        for resolution in [2, 5, 10, 20, 30, 40, 50]:
            binned_cell_data = torch.zeros(
                [data['n_genes'], resolution, resolution]
                )
            for gene in range(data['n_genes']):
                gene_data = data['coordinates'].query('gene=={}'.format(gene))
                binned_cell_data[gene] = utils.binning(
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
                    .reshape(1, -1, resolution, resolution)
                    )
            z_original = z.clone()
            z = torch.nn.functional.interpolate(
                z, [data['res'], data['res']]
                ).squeeze(0)
            w = nmf.components_.T

            results = {
                'z' : z,
                'w' : w,
                'z_original' : z_original,
                'binned_data' : binned_cell_data,
                'intensity_scale' : intensity_scale,
                'res' : resolution,
                'cell' : cell,
            }

            if not os.path.isdir('results/nmf/intensity_scale_{}/cell_{}/'
                    .format(intensity_scale, cell)):
                os.makedirs('results/nmf/intensity_scale_{}/cell_{}/'
                    .format(intensity_scale, cell))
            torch.save(
                results,
                'results/nmf/intensity_scale_{}/cell_{}/resolution_{}.pkl'
                    .format(intensity_scale, cell, resolution)
                )