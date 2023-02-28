import git
import os
import sys
import numpy as np
import pickle
import random
import torch
from anndata import AnnData
import tensorflow as tf
from tensorflow_probability import math as tm
from src import utils

# download code from NSF paper (Townes et al., 2021)
if not os.path.isdir('nsf-paper/'):
    repo = git.Repo.clone_from(
        url='https://github.com/willtownes/nsf-paper',
        to_path='nsf-paper/',
        branch='main',
        no_checkout=True
        )

    repo.git.checkout('928b36c54cb19fae1fa3e86748e2a6846f40210a')

sys.path.insert(1, os.path.join(sys.path[0], 'nsf-paper/'))
from models import sf
from utils import preprocess, misc, training, postprocess

for intensity_scale in [50, 100, 200, 300, 400]:
    for cell in range(20):
        data = torch.load('../../data/simulated/data/independent/{}_{}.pt'
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
            binned_cell_data = binned_cell_data.view(data['n_genes'], -1).T
            binned_cell_data = binned_cell_data.numpy()

            torch.manual_seed(1234)
            random.seed(1234)
            np.random.seed(1234)
            tf.random.set_seed(1234)

            grid = misc.make_grid(binned_cell_data.shape[0])
            grid[:,1] = -grid[:,1]
            grid = preprocess.rescale_spatial_coords(grid)

            ad = AnnData(
                X=binned_cell_data,
                obsm={"spatial" : grid},
            )

            ad.layers = {"counts" : ad.X.copy()}

            D, D_val = preprocess.anndata_to_train_val(ad, layer="counts",
                train_frac=1., flip_yaxis=False)

            n_obs = D['Y'].shape[0]
            n_genes = D['Y'].shape[1]
            grid = D['X']

            D_tf = preprocess.prepare_datasets_tf(D, Dval=D_val)

            n_factors = 3
            inducing_points = misc.kmeans_inducing_pts(grid, 500)
            n_inducing_points = inducing_points.shape[0]
            kernel = tm.psd_kernels.MaternThreeHalves

            try:
                model = sf.SpatialFactorization(n_genes, n_factors, 
                    inducing_points, psd_kernel=kernel, nonneg=True, lik='poi')

                model.init_loadings(D['Y'], X=grid, shrinkage=0.3)
                trainer = training.ModelTrainer(model)
                trainer.train_model(*D_tf, status_freq=50)

                model_interpret = postprocess.interpret_nsf(
                    model, grid, S=100, lda_mode=False
                    )

                z_inferred_original = model_interpret['factors'].T.reshape(
                    [n_factors, resolution, resolution])
                z_inferred_original = torch.tensor(z_inferred_original)
                z_inferred = torch.nn.functional.interpolate(
                    z_inferred_original.unsqueeze(0), [50, 50]).squeeze(0)

                w_inferred = model_interpret['loadings']

                results = {
                    'z' : z_inferred,
                    'z_original' : z_inferred_original,
                    'w' : w_inferred,
                }

                os.makedirs('results/nsf/{}_{}/'
                    .format(intensity_scale, cell), exist_ok=True)
                    
                pickle.dump(
                    results,
                    open('results/nsf/{}_{}/{}.pkl'
                        .format(intensity_scale, cell, resolution), 'wb'))

            except:
                continue