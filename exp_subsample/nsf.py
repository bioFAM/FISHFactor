# Requires NSF implementation from https://github.com/willtownes/nsf-paper

if __name__ == '__main__':
    import os
    import sys
    import pandas as pd
    from anndata import AnnData
    import pickle
    import numpy as np
    from scanpy import pp
    from tensorflow_probability import math as tm
    import tensorflow as tf
    tfk = tm.psd_kernels

    sys.path.insert(1, os.path.join(sys.path[0], '../nsf_paper'))
    from models import pf
    from utils import preprocess, misc, training, postprocess

    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    from src.utils import spatial_binning

    ##################
    ### Parameters ###
    ##################
    n_cells = 20
    subsample_fractions = [0.1, 0.3, 0.5, 0.8, 1.0]
    bin_res_list = [5, 10, 20, 30, 40]

    n_latents = 3
    ###############################
    ###############################

    protrusion_genes = ['Cyb5r3', 'Sh3pxd2a', 'Ddr2', 'Net1', 'Trak2', 'Kif1c', 'Kctd10', 'Dynll2', 'Arhgap11a', 'Gxylt1', 'H6pd', 'Gdf11', 'Dync1li2', 'Palld', 'Ppfia1', 'Naa50', 'Ptgfr', 'Zeb1', 'Arhgap32', 'Scd1']
    nucleus_perinucleus_genes = ['Col1a1', 'Fn1', 'Fbln2', 'Col6a2', 'Bgn', 'Nid1', 'Lox', 'P4hb', 'Aebp1', 'Emp1', 'Col5a1', 'Sdc4', 'Postn', 'Col3a1', 'Pdia6', 'Col5a2', 'Itgb1', 'Calu', 'Pdia3', 'Cyr61']
    cytoplasm_genes = ['Ddb1', 'Myh9', 'Actn1', 'Tagln2', 'Kpnb1', 'Hnrnpf', 'Ppp1ca', 'Hnrnpl', 'Pcbp1', 'Tagln', 'Fscn1', 'Psat1', 'Cald1', 'Snd1', 'Uba1', 'Hnrnpm', 'Cap1', 'Ssrp1', 'Ugdh', 'Caprin1']
    cluster_genes = protrusion_genes + nucleus_perinucleus_genes + cytoplasm_genes

    data = pd.read_feather('../data/nih3t3/preprocessed_data.feather')
    data = data[data.gene.isin(cluster_genes)]

    data['group'] = data['experiment'].astype(str) + '_' + data['fov'].astype(str) + '_' + data['cell'].astype(str)

    for group in data.group.unique()[:n_cells]:
        cell_data = data[data.group==group]
        cell_data = cell_data[['x', 'y', 'gene', 'group']].rename(columns={'gene' : 'feature'})

        # normalize cell coordinates to unit square
        scaling_factor = max(cell_data.x.max() - cell_data.x.min(), cell_data.y.max() - cell_data.y.min())
        cell_data.x = (cell_data.x - cell_data.x.min()) / scaling_factor
        cell_data.y = (cell_data.y - cell_data.y.min()) / scaling_factor

        np.random.seed(1234)
        tf.random.set_seed(1234)

        for subsample_fraction in subsample_fractions:
            for bin_res in bin_res_list:
                if os.path.exists('results/nsf/cell_%s/subsample_%s/binres_%s.pkl'
                    %(group, subsample_fraction, bin_res)):
                    print('File already exists: results/nsf/cell_%s/subsample_%s/binres_%s.pkl'
                        % (group, subsample_fraction, bin_res))
                    continue

                # create subsamples of the data and make sure that every gene occurs at least once
                subsample_size = int(subsample_fraction * len(cell_data))
                first_gene_occurence = cell_data.drop_duplicates('feature')
                other_data = cell_data.drop(index=first_gene_occurence.index)
                subsampled_other_data = other_data.sample(n=subsample_size - len(first_gene_occurence))
                subsampled_data = pd.concat([first_gene_occurence, subsampled_other_data])

                # count points in spatial bins
                subsampled_binned_data = spatial_binning(subsampled_data, nx=bin_res, ny=bin_res).numpy()
                subsampled_binned_data = subsampled_binned_data.reshape(subsampled_binned_data.shape[0], -1).T

                # coordinates
                grid = misc.make_grid(subsampled_binned_data.shape[0])
                grid[:,1] = -grid[:,1] #make the display the same
                grid = preprocess.rescale_spatial_coords(grid)

                # store as AnnData object
                ad = AnnData(
                    X=subsampled_binned_data,
                    obsm={"spatial" : grid}
                )
                ad.layers = {"counts":ad.X.copy()}
                pp.log1p(ad)

                J = subsampled_binned_data.shape[-1] # number of features

                # Preprocessing
                D, _ = preprocess.anndata_to_train_val(
                    ad,
                    layer="counts",
                    train_frac=1.0,
                    flip_yaxis=False
                )

                D_n, _ = preprocess.anndata_to_train_val(
                    ad,
                    train_frac=1.0,
                    flip_yaxis=False
                )

                fmeans, D_c, _ = preprocess.center_data(D_n)
                X = D["X"]
                N = X.shape[0]
                Dtf = preprocess.prepare_datasets_tf(D, Dval=None, shuffle=False)
                Dtf_n = preprocess.prepare_datasets_tf(D_n, Dval=None, shuffle=False)
                Dtf_c = preprocess.prepare_datasets_tf(D_c, Dval=None, shuffle=False)

                L = n_latents
                M = N # number of inducing points
                Z = X # inducing points (use all data points)
                ker = tfk.MaternThreeHalves

                np.random.seed(10)

                fit = pf.ProcessFactorization(J, L, Z, psd_kernel=ker, nonneg=True, lik="poi")
                fit.init_loadings(D["Y"], X=X, sz=D["sz"], shrinkage=0.3)
                tro = training.ModelTrainer(fit)
                try:
                    tro.train_model(*Dtf)
                except ValueError:
                    continue

                inpf = postprocess.interpret_npf(fit, X, S=100, lda_mode=False)

                z_inferred = inpf['factors'].T
                w_inferred = inpf['loadings']

                results = {
                    'z' : z_inferred,
                    'w' : w_inferred,
                }

                if not os.path.exists('results/nsf/cell_%s/subsample_%s/' %(group, subsample_fraction)):
                    os.makedirs('results/nsf/cell_%s/subsample_%s/' %(group, subsample_fraction))

                pickle.dump(results, open('results/nsf/cell_%s/subsample_%s/binres_%s.pkl' %(group, subsample_fraction, bin_res), 'wb'))