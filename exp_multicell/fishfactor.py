if __name__ == '__main__':
    import os
    import sys

    sys.path.insert(1, os.path.join(sys.path[0], '..'))

    import torch
    import pandas as pd
    from src import fishfactor
    import pyro

    ##################
    ### Parameters ###
    ##################
    n_groups = 20
    n_latents = 3
    n_inducing = 50
    grid_resolution = 50
    device = sys.argv[1]
    nu = 1.5
    min_density = 0.1
    normalize_coordinates = True

    lr = 5e-3
    lrd = 1
    n_particles = 15
    max_epochs = 50000
    patience = 2000
    max_points = 10000
    delta = 0.01
    save_every = 1000
    save_dir = 'results_%s/' %n_groups
    max_points = 10000
    ###############################
    ###############################

    data = pd.read_feather('../data/nih3t3/preprocessed_data.feather')

    protrusion_genes = ['Cyb5r3', 'Sh3pxd2a', 'Ddr2', 'Net1', 'Trak2', 'Kif1c', 'Kctd10', 'Dynll2', 'Arhgap11a', 'Gxylt1', 'H6pd', 'Gdf11', 'Dync1li2', 'Palld', 'Ppfia1', 'Naa50', 'Ptgfr', 'Zeb1', 'Arhgap32', 'Scd1']
    nucleus_perinucleus_genes = ['Col1a1', 'Fn1', 'Fbln2', 'Col6a2', 'Bgn', 'Nid1', 'Lox', 'P4hb', 'Aebp1', 'Emp1', 'Col5a1', 'Sdc4', 'Postn', 'Col3a1', 'Pdia6', 'Col5a2', 'Itgb1', 'Calu', 'Pdia3', 'Cyr61']
    cytoplasm_genes = ['Ddb1', 'Myh9', 'Actn1', 'Tagln2', 'Kpnb1', 'Hnrnpf', 'Ppp1ca', 'Hnrnpl', 'Pcbp1', 'Tagln', 'Fscn1', 'Psat1', 'Cald1', 'Snd1', 'Uba1', 'Hnrnpm', 'Cap1', 'Ssrp1', 'Ugdh', 'Caprin1']
    cluster_genes = protrusion_genes + nucleus_perinucleus_genes + cytoplasm_genes

    data = data[data.gene.isin(cluster_genes)]
    data['group'] = data['experiment'].astype(str) + '_' + data['fov'].astype(str) + '_' + data['cell'].astype(str)
    data = data.rename(columns={'gene' : 'feature'})
    data = data[['x', 'y', 'feature', 'group']]

    groups = data.group.unique()[:n_groups]
    data = data[data.group.isin(groups)]

    torch.manual_seed(1234)
    pyro.clear_param_store()

    model = fishfactor.FISHFactor(
        data=data,
        n_latents=n_latents,
        nu=nu,
        n_inducing=n_inducing,
        grid_resolution=grid_resolution,
        min_density=min_density,
        normalize_coordinates=normalize_coordinates,
        device=device,
    ).to(device=device)

    results = model.inference(
        lr=lr,
        lrd=lrd,
        n_particles=n_particles,
        max_epochs=max_epochs,
        patience=patience,
        delta=delta,
        save_every=save_every,
        save_dir=save_dir,
        max_points=max_points,
        )

    torch.save(results, os.path.join(save_dir, 'results_final.pkl'))