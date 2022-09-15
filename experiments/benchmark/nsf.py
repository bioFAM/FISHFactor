# requires https://github.com/willtownes/nsf-paper.git in a subdirectory nsf_paper/
# and the required packages, e.g. anndata, tensorflow, tensorflow_probability
import os
import sys
from anndata import AnnData
import pickle
from tensorflow_probability import math as tm
import numpy as np
import tensorflow as tf
import random
import torch
tfk = tm.psd_kernels

sys.path.insert(1, os.path.join(sys.path[0], 'nsf_paper/'))
from models import sf
from utils import preprocess, misc, training, postprocess

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from src.utils import binning

for intensity_scale in [50, 100, 200, 300]:
    for cell in range(20):
        data = torch.load('data/intensity_scale_{}/cell_{}.pkl'
            .format(intensity_scale, cell))

        for resolution in [2, 5, 10, 20, 30, 40, 50]:
            random.seed(1234)
            np.random.seed(1234)
            tf.random.set_seed(1234)

            binned_cell_data = torch.zeros(
                [data['n_genes'], resolution, resolution]
                )
            for gene in range(data['n_genes']):
                gene_data = data['coordinates'].query('gene=={}'.format(gene))
                binned_cell_data[gene] = binning(
                    torch.tensor(gene_data.x.values),
                    torch.tensor(gene_data.y.values),
                    xlim=[0, 1],
                    ylim=[0, 1],
                    resolution=resolution,
                    )
            binned_cell_data = binned_cell_data.view(data['n_genes'], -1).T
            binned_cell_data = binned_cell_data.numpy()

            grid = misc.make_grid(binned_cell_data.shape[0])
            grid[:,1] = -grid[:,1]
            grid = preprocess.rescale_spatial_coords(grid)

            ad = AnnData(
                X=binned_cell_data,
                obsm={"spatial" : grid},
            )
            ad.layers = {"counts" : ad.X.copy()}

            D, D_val = preprocess.anndata_to_train_val(
                ad, layer="counts", train_frac=1., flip_yaxis=False
                )

            n_obs = D['Y'].shape[0]
            n_genes = D['Y'].shape[1]
            grid = D['X']

            D_tf = preprocess.prepare_datasets_tf(D, Dval=D_val)

            n_factors = 3
            inducing_points = misc.kmeans_inducing_pts(grid, 500)
            n_inducing_points = inducing_points.shape[0]
            kernel = tfk.MaternThreeHalves

            try:
                model = sf.SpatialFactorization(
                    n_genes, n_factors, inducing_points, psd_kernel=kernel,
                    nonneg=True, lik='poi'
                    )
                model.init_loadings(D['Y'], X=grid, shrinkage=0.3)
                trainer = training.ModelTrainer(model)
                trainer.train_model(*D_tf, status_freq=50)

                model_interpret = postprocess.interpret_nsf(
                    model, grid, S=100, lda_mode=False
                    )

                z_inferred_original = model_interpret['factors'].T.reshape(
                    [n_factors, resolution, resolution]
                    )
                z_inferred_original = torch.tensor(z_inferred_original)
                z_inferred = torch.nn.functional.interpolate(
                    z_inferred_original.unsqueeze(0),
                    [50, 50],
                    ).squeeze(0)

                w_inferred = model_interpret['loadings']

                results = {
                    'z' : z_inferred,
                    'z_original' : z_inferred_original,
                    'w' : w_inferred,
                }

                os.makedirs('results/nsf/intensity_scale_{}/cell_{}'
                    .format(intensity_scale, cell), exist_ok=True)
                pickle.dump(
                    results,
                    open('results/nsf/intensity_scale_{}/cell_{}/resolution_{}.pkl'
                        .format(intensity_scale, cell, resolution), 'wb'))

            except:
                continue