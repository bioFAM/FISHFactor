if __name__ == '__main__':
    import os
    import sys

    sys.path.insert(1, os.path.join(sys.path[0], '..'))

    import pandas as pd
    import pyro
    import torch
    import pickle
    from src import fishfactor

    ##################
    ### Parameters ###
    ##################
    factor_set_list = [i for i in range(10)]
    intensity_scale_list = [20, 50, 80, 100, 150, 200]
    n_latents = 3
    n_inducing = 100
    nu = 1.5
    grid_resolution = 50
    min_density = 0.1
    device = sys.argv[1]

    lr = 5e-3
    lrd = 0.5**(1 / 1000)
    n_particles = 15
    max_epochs = 15000
    patience = 2500
    delta = 0.01
    normalize_coordinates = False
    max_points = 10000
    ###############################
    ###############################

    for factor_set in factor_set_list:
        for intensity_scale in intensity_scale_list:
            if os.path.exists('results/fishfactor/factors_%s/intensity_%s.pkl'
                %(factor_set, intensity_scale)):
                print('File already exists: results/fishfactor/factors_%s/intensity_%s.pkl'
                %(factor_set, intensity_scale))
                continue

            data = pickle.load(open('../data/pp_sim/data/factors_%s/intensity_%s.pkl'
                %(factor_set, intensity_scale), 'rb'))
            df = pd.DataFrame(data['data'].numpy(), columns=['x', 'y', 'feature'])
            df['group'] = 0

            pyro.clear_param_store()
            torch.manual_seed(1234)

            model = fishfactor.FISHFactor(
                data=df,
                n_latents=n_latents,
                n_inducing=n_inducing,
                nu=nu,
                grid_resolution=grid_resolution,
                min_density=min_density,
                device=device,
                normalize_coordinates=normalize_coordinates,
            ).to(device=device)

            results = model.inference(
                lr=lr,
                lrd=lrd,
                n_particles=n_particles,
                max_epochs=max_epochs,
                patience=patience,
                delta=delta,
                max_points=max_points,
            )

            if not os.path.exists('results/fishfactor/factors_%s/'
                %factor_set):
                os.makedirs('results/fishfactor/factors_%s'
                % factor_set)

            torch.save(results, 'results/fishfactor/factors_%s/intensity_%s.pkl'
                %(factor_set, intensity_scale))