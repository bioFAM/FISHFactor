import typing
import os
import torch
import pandas as pd
import numpy as np
import pyro
import gpytorch
import time
from src import utils


class BatchGP(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points: torch.tensor,
        nu: float=1.5,
        use_scale_factor=True,
        ):
        """Batch svGP in 2D with Matérn kernel.
        
        :param inducing_points: tensor of shape (batch_size, n_inducing, 2) with
            initial inducing coordinates (determines batch size)
        :param nu: smoothness parameter of Matérn kernel, can be 0.5, 1.5, 2.5
        """
        batch_shape = inducing_points.shape[0]

        variational_distribution = (
            gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[1],
            batch_shape=torch.Size([batch_shape]),
            )
        )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            model=self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
        )

        super().__init__(variational_strategy=variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size([batch_shape]),
        )

        if use_scale_factor:
            base_covar_module = gpytorch.kernels.MaternKernel(
                batch_shape=torch.Size([batch_shape]),
                nu=nu,
            )

            self.covar_module = gpytorch.kernels.ScaleKernel(
                base_covar_module,
                batch_shape=torch.Size([batch_shape])
            )
        else:
            self.covar_module = gpytorch.kernels.MaternKernel(
                batch_shape=torch.Size([batch_shape]),
                nu=nu,
            )

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class FISHFactor(gpytorch.Module):
    def __init__(
        self,
        data: pd.DataFrame,
        n_latents: int,
        nu: float=1.5,
        n_inducing: int=50,
        grid_resolution: int=50,
        min_density: float=0.1,
        masks: typing.Optional[torch.tensor]=None,
        normalize_coordinates: bool=True,
        device: str='cpu',
        ):
        """FISHFactor model.
        
        :param data: DataFrame with columns 'x', 'y', 'feature', 'group'
        :param n_latents: number of latent processes per group
        :param nu: smoothness parameter of Matérn kernel
        :param n_inducing: number of inducing points per latent process
        :param grid_resolution: resolution per spatial dimension of a discrete
            grid that is used e.g. to compute quadrature values in the 
            likelihood integral and to create a binary mask of the cell
        :param min_density: minimum value of the kernel density estimate of
            points in every group to be included in the group mask
        :param masks: replaces masks created by density estimate, if specified
        :param normalize_coordinates: if True, rescale points of every
            group to the unit square
        :param device: 'cpu' or 'cuda:x'
        """
        super().__init__()

        self.K = n_latents
        self.n_inducing = n_inducing
        self.grid_resolution = grid_resolution
        self.device = device
        self.nu = nu
        self.min_density = min_density

        # enumerate features
        data['feature_id'], self.features = pd.factorize(data['feature'])
        self.D = len(self.features) # number of features

        # enumerate groups
        data['group_id'], self.groups = pd.factorize(data['group'])
        self.M = len(self.groups) # number of groups

        # number of data points per group
        self.N_m = torch.tensor(data.groupby('group_id').size())

        self.data_tensor = torch.tensor(
            data=data[['x', 'y', 'feature_id', 'group_id']].values,
            dtype=torch.float32,
            device=self.device
        )

        # group-wise rescaling of coordinates to unit square
        if normalize_coordinates:
            self.data_tensor = utils.normalize_group_coordinates(
                self.data_tensor)

        # regularly spaced grid in unit square
        grid = utils.grid(grid_resolution)
        self.grid = grid.to(dtype=torch.float32, device=self.device)

        # mask for every group, based on threshold on kernel density estimate
        # of data points in the group
        if masks == None:
            masks = []
            for group in self.data_tensor[:, 3].unique():
                masks.append(
                    utils.density_mask(
                        data=self.data_tensor[self.data_tensor[:, 3]==group],
                        grid=self.grid,
                        min_density=min_density,
                )
            )
            self.masks = torch.stack(masks, dim=0).to(device=self.device)
        else:
            self.masks = masks.to(device=self.device)

        # average intensity per feature and group, given by number of points
        # divided by group mask area
        cell_area = self.masks.sum(dim=[-1, -2]) / self.grid_resolution**2
        n_points = data.groupby(['feature_id', 'group_id']).size().reset_index()
        table = n_points.pivot('group_id', 'feature_id', 0).fillna(0).values.T
        mean_intensity = torch.tensor(table) / cell_area.view(1, self.M).cpu()
        self.mean_intensity = mean_intensity.to(
            dtype=torch.float32,
            device=self.device
        )

        # keep track of the groups that were subsampled
        self.m_inds = []

        # instantiate one batch gp per group
        if self.M == 1:
            use_scale_factor = False
        else:
            use_scale_factor = True

        self.gp_list = []            
        for m in range(self.M):
            # choose random data points as initial inducing locations
            group_inds = (self.data_tensor[:, 3] == m).cpu()
            inducing_points = torch.zeros([self.K, n_inducing, 2])
            for k in range(self.K):
                random_inds = np.random.choice(
                    a=group_inds.nonzero().flatten(),
                    size=[n_inducing]
                )
                random_points = self.data_tensor[random_inds].clone().detach()
                inducing_points[k] = random_points[:, [0, 1]]

            self.gp_list.append(BatchGP(
                inducing_points=inducing_points,
                nu=nu,
                use_scale_factor=use_scale_factor,
            ).to(device=self.device))
            
    def model(
        self,
        x: torch.tensor,
        subsample_inds: typing.Sequence
        ):
        """Pyro model.
        
        :param x: tensor of shape (N, 4) with x coordinate in first, y
            coordinate in second, numeric feature value in third and numeric
            group value in fourth column.
        :param subsample_inds: list of index tensors for data points in every
            group, can be subsampled or not.
        """
        for m in range(self.M):
            pyro.module('gp_%s' %m, self.gp_list[m])
        
        N_plate = pyro.plate('N_plate', dim=-1, device=self.device)
        K_plate = pyro.plate('K_plate', dim=-2, device=self.device, size=self.K)
        M_plate = pyro.plate('M_plate', dim=-3, device=self.device, size=self.M)
        D_plate = pyro.plate('D_plate', dim=-4, device=self.device, size=self.D)

        # weights
        with D_plate, K_plate:
            w = pyro.sample(
                name='w',
                fn=pyro.distributions.Normal(
                    loc=torch.tensor(
                        data=0.,
                        dtype=torch.float32,
                        device=self.device
                    ),
                    scale=torch.tensor(
                        data=1.,
                        dtype=torch.float32,
                        device=self.device
                    ),
                ),
            ).view(-1, self.D, 1, self.K, 1)
        w = torch.nn.Softplus()(w)

        # factors
        with M_plate as m_ind, K_plate, N_plate:
            group_data = x[x[:, 3]==m_ind[0]]
            subsampled_data = group_data[subsample_inds[m_ind[0]].bool()]
            eval_points = torch.cat([subsampled_data[:, [0, 1]], self.grid])
            z = pyro.sample(
                name='z',
                fn=self.gp_list[m_ind[0]].pyro_model(eval_points),
            ).view(-1, 1, 1, self.K, eval_points.shape[0])
        z = torch.nn.Softplus()(z)

        # intensity function
        intensity = torch.matmul(
            input=w.transpose(-1, -2),
            other=z
        ).view(-1, self.D, 1, 1, eval_points.shape[0])

        # scale intensity function with average intensity per feature in the
        # subsampled group and with the data subsampling fraction p to get an
        # estimate on the correct scale
        group_mean_intensity = self.mean_intensity[:, m_ind[0]]
        group_mean_intensity = group_mean_intensity.view(1, self.D, 1, 1, 1)
        intensity = intensity * group_mean_intensity * self.p[m_ind[0]]

        # intensity values at data and quadrature points
        data_intensity, quadrature_intensity = intensity.split(
            split_size=[subsampled_data.shape[0], self.grid.shape[0]],
            dim=-1,
        )

        # set quadrature intensity values at background points to zero
        group_mask = self.masks[m_ind[0]].view(1, 1, 1, 1, self.grid.shape[0])
        masked_quadrature_intensity = quadrature_intensity * group_mask
        
        # Poisson point process likelihood
        expectation = (
            masked_quadrature_intensity.sum(dim=-1) / group_mask.sum()
        ).sum(dim=[1, 2, 3])

        point_log_intensity = (
            (data_intensity[:, subsampled_data[:, 2].long(), 0, 0,
                torch.arange(subsampled_data.shape[0])]).log().sum(dim=-1))

        pyro.factor('log_likelihood', point_log_intensity - expectation)

    def guide(
        self,
        x: torch.tensor,
        subsample_inds: typing.Sequence,
        ):
        """Pyro guide.
        
        for parameters see model.
        """
        N_plate = pyro.plate('N_plate', dim=-1, device=self.device)
        K_plate = pyro.plate('K_plate', dim=-2, device=self.device, size=self.K)
        M_plate = pyro.plate('M_plate', dim=-3, device=self.device, size=self.M,
            subsample_size=1)
        D_plate = pyro.plate('D_plate', dim=-4, device=self.device, size=self.D)

        # weights
        w_loc_raw = pyro.param(
            name='w_loc_raw',
            init_tensor=torch.full(
                size=[self.D, 1, self.K, 1],
                fill_value=0.0,
                dtype=torch.float32,
                device=self.device
            ),
        )

        w_scale_raw = pyro.param(
            name='w_scale_raw',
            init_tensor=torch.full(
                size=[self.D, 1, self.K, 1],
                fill_value=1.0,
                dtype=torch.float32,
                device=self.device
            ),
            constraint=pyro.distributions.constraints.positive,
        )

        with D_plate, K_plate:
            pyro.sample(
                name='w',
                fn=pyro.distributions.Normal(w_loc_raw, w_scale_raw),
                infer=dict(baseline={'use_decaying_avg_baseline' : True}),
            )

        # factors
        with M_plate as m_ind, K_plate, N_plate:
            self.m_inds.append(m_ind[0])
            group_data = x[x[:, 3]==m_ind[0]]
            subsampled_data = group_data[subsample_inds[m_ind[0]].bool()]
            eval_points = torch.cat([subsampled_data[:, [0, 1]], self.grid])
            pyro.sample('z', self.gp_list[m_ind[0]].pyro_guide(eval_points))

    def inference(
        self,
        lr: float=5e-3,
        lrd: float=1.,
        n_particles: int=15,
        max_epochs: int=10000,
        patience: int=1000,
        delta: float=0.01,
        save_every: typing.Optional[int]=None,
        save_dir: typing.Optional[str]=None,
        max_points: int=5000,
        print_every: int=100,
        ):
        """Do inference.

        :param lr: initial learning rate
        :param lrd: learning rate decrease factor
        :param n_particles: number of samples to use for ELBO gradient estimate
        :param max_epochs: maximum number of epochs before training terminates
        :param patience: number of epochs without relevant loss improvement
            before training is stopped
        :param delta: minimum absolute loss decrease to count as improvement
        :param save_every: number of epochs after which model state is saved
        :param save_dir: directory where intermediate states are saved
        :param max_points: maximum number of data points to use, otherwise
            do subsampling to use less memory
        :param print_every: number of epochs after which loss is printed
        """
        self.lr = lr
        self.lrd = lrd
        self.n_particles = n_particles
        self.max_epochs = max_epochs
        self.patience = patience
        self.delta = delta
        self.save_every = save_every
        self.save_dir = save_dir
        self.max_points = max_points

        # make directory to save results, if it does not exist yet
        if save_dir != None:
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

        optimizer = pyro.optim.ClippedAdam({'lr' : lr, 'lrd' : lrd})

        elbo = pyro.infer.Trace_ELBO(
            retain_graph=True,
            num_particles=n_particles,
            vectorize_particles=True,
        )

        svi = pyro.infer.SVI(
            model=self.model,
            guide=self.guide,
            optim=optimizer,
            loss=elbo
        )

        self.train()

        self.p = torch.clamp(max_points / self.N_m, max=1.)

        subsample_inds = []
        for m in range(self.M):
            subsample_inds.append(
                torch.bernoulli(
                    torch.full(size=[self.N_m[m]], fill_value=self.p[m])
                )
            )
        init_loss = svi.evaluate_loss(self.data_tensor, subsample_inds)
        min_loss = torch.full([self.M], 1e5)
        wait_epochs = 0
        loss = []

        if self.device.startswith('cuda'):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start_time = time.time()

        # optimization loop
        for epoch in range(max_epochs):
            subsample_inds = []
            for m in range(self.M):
                subsample_inds.append(
                    torch.bernoulli(
                        torch.full(size=[self.N_m[m]], fill_value=self.p[m])
                    )
                )
            with pyro.poutine.scale(scale=1.0 / abs(init_loss)):
                self.zero_grad()
                loss.append(svi.step(self.data_tensor, subsample_inds))

            if epoch % print_every == 0:
                print(
                    'epoch: %s, loss: %s, min loss: %s, patience: %s'
                    %(epoch, round(loss[-1], 4), min_loss.tolist(), patience - wait_epochs)
                )

            # early stopping
            if (loss[-1] <= (min_loss[self.m_inds[-1]] - delta)) & (epoch > 10):
                min_loss[self.m_inds[-1]] = loss[-1]
                wait_epochs = 0
            else:
                wait_epochs += 1
            
            if wait_epochs > patience:
                break

            # save
            if save_every != None:
                if epoch % save_every == 0:
                    gp_state_dicts = []
                    for m in range(self.M):
                        gp_state_dicts.append(self.gp_list[m].state_dict())
                        
                    results = {
                        'pyro_params' : dict(pyro.get_param_store()),
                        'state_dict' : self.state_dict(),
                        'gp_state_dicts' : gp_state_dicts,
                        'loss' : loss,
                        'init_loss' : init_loss,
                        'p' : self.p,
                    }

                    torch.save(results, os.path.join(
                        save_dir, 'results_%s_epochs.pkl' %epoch)
                    )

        if self.device.startswith('cuda'):
            end.record()
            torch.cuda.synchronize()
            runtime = start.elapsed_time(end)
        else:
            stop_time = time.time()
            runtime = stop_time - start_time

        gp_state_dicts = []
        for m in range(self.M):
            gp_state_dicts.append(self.gp_list[m].state_dict())
        
        results = {
            'pyro_params' : dict(pyro.get_param_store()),
            'state_dict' : self.state_dict(),
            'gp_state_dicts' : gp_state_dicts,
            'loss' : loss,
            'init_loss' : init_loss,
            'p' : self.p,
            'runtime' : runtime,
        }

        return results
