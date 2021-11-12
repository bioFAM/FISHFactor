# requires the PNMF implementation from https://github.com/willtownes/nsf-paper

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

    np.random.seed(1234)
    tf.random.set_seed(1234)

    sys.path.insert(1, os.path.join(sys.path[0], '../nsf_paper'))
    from models import cf
    from utils import preprocess, misc, training, postprocess

    sys.path.insert(1, os.path.join(sys.path[0], '..'))
    from src.utils import spatial_binning

    ##################
    ### Parameters ###
    ##################
    bin_res_list = [5, 10, 20, 30, 40]
    factor_set_list = [i for i in range(10)]
    intensity_scale_list = [20, 50, 80, 100, 150]
    n_latents = 3
    ###############################
    ###############################

    for bin_res in bin_res_list:
        for factor_set in factor_set_list:
            for intensity_scale in intensity_scale_list:
                if os.path.exists('results/pnmf_paper/factors_%s/intensity_%s/binres_%s.pkl'
                    %(factor_set, intensity_scale, bin_res)):
                    print('File already exists: results/pnmf_paper/factors_%s/intensity_%s/binres_%s.pkl'
                    %(factor_set, intensity_scale, bin_res))
                    continue

                data = pickle.load(open('../data/pp_sim/data/factors_%s/intensity_%s.pkl'
                    %(factor_set, intensity_scale), 'rb'))

                # bin counts
                df = pd.DataFrame(data['data'].numpy())
                df.columns = ['x', 'y', 'feature']

                binned_data = spatial_binning(df, nx=bin_res, ny=bin_res).numpy()
                binned_data = binned_data.reshape(binned_data.shape[0], -1).T

                # coordinates
                grid = misc.make_grid(binned_data.shape[0])
                grid[:,1] = -grid[:,1] #make the display the same
                grid = preprocess.rescale_spatial_coords(grid)

                # store as AnnData object
                ad = AnnData(
                    X=binned_data,
                    obsm={"spatial" : grid},
                )
                ad.layers = {"counts":ad.X.copy()}
                pp.log1p(ad)

                J = binned_data.shape[-1] # number of features

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
                Dtf = preprocess.prepare_datasets_tf(D, Dval=None,
                    shuffle=False)
                Dtf_n = preprocess.prepare_datasets_tf(D_n, Dval=None,
                    shuffle=False)
                Dtf_c = preprocess.prepare_datasets_tf(D_c, Dval=None,
                    shuffle=False)

                L = n_latents
                M = N # number of inducing points
                Z = X # inducing points (use all data points)
                ker = tfk.MaternThreeHalves

                fit = cf.CountFactorization(N, J, L, nonneg=True, lik="poi")
                fit.init_loadings(D["Y"], sz=D["sz"], shrinkage=0.3)
                tro = training.ModelTrainer(fit)
                try:
                    tro.train_model(*Dtf)
                except ValueError:
                    continue
                incf = postprocess.interpret_ncf(fit, S=100, lda_mode=False)

                z_inferred = incf['factors'].T
                w_inferred = incf['loadings']

                results = {
                    'z' : z_inferred,
                    'w' : w_inferred,
                }

                if not os.path.exists('results/pnmf_paper/factors_%s/intensity_%s/'
                    %(factor_set, intensity_scale)):
                    os.makedirs('results/pnmf_paper/factors_%s/intensity_%s/'
                    %(factor_set, intensity_scale))

                pickle.dump(results, open('results/pnmf_paper/factors_%s/intensity_%s/binres_%s.pkl'
                    %(factor_set, intensity_scale, bin_res), 'wb'))