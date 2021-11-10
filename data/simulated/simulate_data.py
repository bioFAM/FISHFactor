if __name__ == '__main__':
    import sys, os
    sys.path.insert(1, os.path.join(sys.path[0], '../..'))
    from src import simulation
    import torch
    import pickle

    intensity_scales=[20, 50, 80, 100, 150, 200, 300]
    factor_sets = [i for i in range(10)]

    for factor_set in factor_sets:
        for intensity_scale in intensity_scales:
            if not os.path.exists('data/factors_%s/' %factor_set):
                os.makedirs('data/factors_%s/' %factor_set)

            print('Generating dataset with intensity scale %s and factor set %s'
                %(intensity_scale, factor_set))
            torch.manual_seed(factor_set)

            data = simulation.simulate_pp(
                n_features=60,
                factor_dir='factors/set%s' %factor_set,
                weight_scales_raw=[1., 1., 1.],
                weight_sparsities=[0.6, 0.8, 0.7],
                sigma=2.,
                intensity_scale=intensity_scale,
            )

            pickle.dump(data, open('data/factors_%s/intensity_%s.pkl'
                %(factor_set, intensity_scale), 'wb'))