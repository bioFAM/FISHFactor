if __name__ == '__main__':
    import os
    import sys

    sys.path.insert(1, os.path.join(sys.path[0], '..'))

    import pandas as pd
    import pyro
    import torch
    import numpy as np
    from src import fishfactor
    from src import utils

    ##################
    ### Parameters ###
    ##################
    n_cells = 20
    subsample_fractions = [0.1, 0.3, 0.5, 0.8, 1.0]
    
    n_latents = 3
    nu = 1.5
    n_inducing = 100
    grid_resolution = 50
    min_density = 0.1
    normalize_coordinates = False
    device = sys.argv[1]

    lr = 5e-3
    lrd = 0.5**(1 / 1000)
    n_particles = 15
    max_epochs = 10000
    patience = 2000
    delta = 0.01
    save_every = 1000
    max_points = 10000

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
        cell_data['feature_id'], _ = pd.factorize(cell_data.feature)
        cell_data['group_id'], _ = pd.factorize(cell_data.group)

        # normalize cell coordinates to unit square
        scaling_factor = max(cell_data.x.max() - cell_data.x.min(), cell_data.y.max() - cell_data.y.min())
        cell_data.x = (cell_data.x - cell_data.x.min()) / scaling_factor
        cell_data.y = (cell_data.y - cell_data.y.min()) / scaling_factor

        cell_data_tensor = torch.tensor(cell_data[['x', 'y', 'feature_id', 'group_id']].values)
        grid = utils.grid(grid_resolution)
        mask = utils.density_mask(cell_data_tensor, grid, min_density).unsqueeze(0)

        for subsample_fraction in subsample_fractions:
            save_dir = 'results/fishfactor/cell_%s/subsample_%s/' %(group, subsample_fraction)
            if os.path.exists('results/fishfactor/cell_%s/subsample_%s.pkl'
                %(group, subsample_fraction)):
                print('File already exists: results/fishfactor/cell_%s/subsample_%s.pkl'
                    % (group, subsample_fraction))
                continue

            torch.manual_seed(1234)
            np.random.seed(1234)
            
            # create subsamples of the data and make sure that every gene occurs at least once
            subsample_size = int(subsample_fraction * len(cell_data))
            first_gene_occurence = cell_data.drop_duplicates('feature')
            other_data = cell_data.drop(index=first_gene_occurence.index)
            subsampled_other_data = other_data.sample(n=subsample_size - len(first_gene_occurence))
            subsampled_data = pd.concat([first_gene_occurence, subsampled_other_data])
            subsampled_data = subsampled_data[['x', 'y', 'feature', 'group']]

            pyro.clear_param_store()

            model = fishfactor.FISHFactor(
                data=subsampled_data,
                n_latents=n_latents,
                n_inducing=n_inducing,
                grid_resolution=grid_resolution,
                device=device,
                normalize_coordinates=normalize_coordinates,
                min_density=min_density,
                masks=mask,
            ).to(device=device)

            results = model.inference(
                lr=lr,
                lrd=lrd,
                n_particles=n_particles,
                max_epochs=max_epochs,
                patience=patience,
                delta=delta,
                max_points=max_points,
                save_every=save_every,
                save_dir=save_dir,
            )

            if not os.path.exists('results/fishfactor/cell_%s/' % group):
                os.makedirs('results/fishfactor/cell_%s/' % group)

            torch.save(results, 'results/fishfactor/cell_%s/subsample_%s.pkl'
                % (group, subsample_fraction))