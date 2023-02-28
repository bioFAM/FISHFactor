import os
import torch
import time
import pyro
import gpytorch
import json
import pickle
import pandas as pd
import numpy as np
from src import utils, gp


class FISHFactor(gpytorch.Module):
    def __init__(
        self,
        data: pd.DataFrame,
        n_factors: int,
        device: str='cpu',
        n_inducing: int=100,
        grid_res: int=50,
        factor_smoothness: float=1.5,
        masks_threshold: float=0.4,
        init_bin_res: int=5,
        ) -> None:
        """FISHFactor model
        
        Args:
            data: DataFrame with 'x', 'y', 'cell', 'gene' columns
                the gene column should have data type str and the cell column
                data type float or int
            n_factors: number of latent spatial factors per cell
            device: 'cpu' or 'cuda:x'
            n_inducing: number of GP inducing points per latent spatial factor
            grid_res: grid resolution, for cell masks and integral quadrature
                in the likelihood term
            factor_smoothness: smoothness parameter of Matérn kernel
                (0.5, 1.5, or 2.5)
            masks_threshold: KDE threshold for cell masks
            init_bin_res: weight initialization resolution (binning based NMF)
        """
        super().__init__()
        pyro.clear_param_store()

        self.data = data
        self.n_factors = n_factors
        self.device = device
        self.n_inducing = n_inducing
        self.grid_res = grid_res
        self.factor_smoothness = factor_smoothness
        self.masks_threshold = masks_threshold
        self.init_bin_res = init_bin_res
        self.to(device=device)

        # sort data
        self.data = self.data.sort_values(['cell', 'gene'])

        # indexing of genes
        self.data['gene_id'], self.genes = pd.factorize(self.data['gene'])
        self.n_genes = len(self.genes)

        # indexing of cells
        self.data['cell_id'], self.cells = pd.factorize(self.data['cell'])
        self.n_cells = len(self.cells)

        # coordinate rescaling of individual cells to the unit square
        self.data = utils.rescale_cells(self.data)

        # grid coordinates in unit square
        self.grid = utils.grid(grid_res).to(device=device)

        # cell masks based on KDE threshold
        self.masks = utils.cell_masks(
            data=self.data,
            grid=self.grid.cpu(),
            threshold=masks_threshold,
            ).to(device=device)

        # initialization of weights with binning based NMF
        self.w_init = utils.initialize_weights(
            data=self.data,
            n_factors=n_factors,
            bin_res=init_bin_res,
            ).to(device=device).detach()
        self.q_init = (self.w_init.exp() - 1).log().clamp(min=-10)

        # determination of scaling factor in likelihood term
        self.scaling = utils.average_intensity(
            data=self.data,
            masks=self.masks.cpu(),
            per_gene=True,
            per_cell=True,
        ).to(device=device)

        # instantiate spatial factors
        self.factor_list = []
        self.inducing_points = []
        for cell_id in range(self.n_cells):
            # sample initial inducing point locations from molecule locations
            cell_data = self.data[self.data.cell_id==cell_id]
            self.inducing_points.append(cell_data.sample(
                n=min(n_inducing, len(cell_data))
                ))

            self.factor_list.append(gp.SpatialFactors(
                inducing_points=self.inducing_points[-1],
                n_factors=n_factors,
                smoothness=factor_smoothness,
                scale_param=self.n_cells > 1,
            ).to(device=device))

    def model(
        self,
        data: torch.tensor,
        cell: int,
        ) -> None:
        """Pyro model.
        
        Args:
            data: tensor of shape (N, 3) with x, y, gene_id columns
            cell: id of selected cell (only one at a time)
        """
        # register GPs in Pyro
        pyro.module('factors_%s' %cell, self.factor_list[cell])

        data_plate = pyro.plate('data_plate', dim=-1, device=self.device)
        factor_plate = pyro.plate('factor_plate', dim=-2, device=self.device,
            size=self.n_factors)
        gene_plate = pyro.plate('gene_plate', dim=-3, device=self.device,
            size=self.n_genes)

        # weights
        with gene_plate, factor_plate:
            q = pyro.sample(
                name='q',
                fn=pyro.distributions.Normal(
                    loc=torch.tensor(0, dtype=torch.float32,
                        device=self.device),
                    scale=torch.tensor(1., dtype=torch.float32,
                        device=self.device),
                ),
            ).view(-1, self.n_genes, self.n_factors, 1)
        w = torch.nn.Softplus()(q)

        # factors
        cell_grid = self.grid[self.masks[cell].flatten()]
        eval_points = torch.cat([data[:, [0, 1]], cell_grid])
        with factor_plate, data_plate:
            f = pyro.sample(
                name='f',
                fn=self.factor_list[cell].pyro_model(eval_points),
            ).view(-1, 1, self.n_factors, eval_points.shape[0])
        z = torch.nn.Softplus()(f)

        # intensity function
        intensity = torch.matmul(
            input=w.squeeze(-1),
            other=z.squeeze(-3),
        ).view(-1, self.n_genes, 1, eval_points.shape[0])

        intensity *= self.scaling[:, cell].view(1, self.n_genes, 1, 1)

        # intensity values at data and quadrature points
        data_intensity, quadrature_intensity = intensity.split(
            split_size=[data.shape[0], cell_grid.shape[0]],
            dim=-1,
        )

        # Poisson point process likelihood
        expectation = (
            quadrature_intensity.sum(dim=-1) / self.masks[cell].sum()
        ).sum(dim=[1, 2,])

        point_log_intensity = (
            (data_intensity[:, data[:, 2].long(), 0,
                torch.arange(data.shape[0])]).log().sum(dim=-1))

        pyro.factor('log_likelihood', point_log_intensity - expectation)

    def guide(
        self,
        data: torch.tensor,
        cell: int,
        ) -> None:
        """Pyro guide.
        
        Args:
            data: tensor of shape (N, 3) with x, y, gene_id columns
            cell: id of selected cell (only one at a time)
        """
        data_plate = pyro.plate('data_plate', dim=-1, device=self.device)
        factor_plate = pyro.plate('factor_plate', dim=-2, device=self.device,
            size=self.n_factors)
        gene_plate = pyro.plate('gene_plate', dim=-3, device=self.device,
            size=self.n_genes)

        # weights
        q_loc = pyro.param(
            name='q_loc',
            init_tensor=(self.q_init
                .view(self.n_genes, self.n_factors, 1).clone())
        )

        q_scale = pyro.param(
            name='q_scale',
            init_tensor=torch.full(
                size=[self.n_genes, self.n_factors, 1],
                fill_value=1.0,
                dtype=torch.float32,
                device=self.device
            ),
            constraint=pyro.distributions.constraints.positive,
        )

        with gene_plate, factor_plate:
            pyro.sample(
                name='q',
                fn=pyro.distributions.Normal(q_loc, q_scale),
                infer=dict(baseline={'use_decaying_avg_baseline' : True}),
            )

        # factors
        with factor_plate, data_plate:
            cell_grid = self.grid[self.masks[cell].flatten()]
            eval_points = torch.cat([data[:, [0, 1]], cell_grid])
            pyro.sample('f', self.factor_list[cell].pyro_guide(eval_points))

    def inference(
        self,
        lr: float=5e-3,
        lrd: float=1.,
        n_particles: int=15,
        early_stopping_patience: int=100,
        min_improvement: float=0.01,
        max_epochs: int=10000,
        save: bool=True,
        save_every: int=100,
        save_dir: str='results/'
        ):
        """Perform stochastic variational inference.
        
        Args:
            lr: adam learning rate
            lrd: learning rate decrease factor
            n_particles: number of samples for ELBO gradient estimate
            early_stopping_patience: number of epochs without improvement
                before training terminates
            min_improvement: minimum absolute ELBO increase to
                be considered an improvement
            max_epochs: maximum number of epochs before training terminates
            save: whether the model is saved
            save_every: number of epochs after which model is saved
            save_dir: where the model is saved
        """
        data_tensor = torch.tensor(
            data=self.data[['x', 'y', 'gene_id', 'cell_id']].values,
            dtype=torch.float32,
            device=self.device,
        ).detach()

        self.optimizer = pyro.optim.ClippedAdam({'lr' : lr, 'lrd' : lrd})

        elbo = pyro.infer.Trace_ELBO(
            retain_graph=True,
            num_particles=n_particles,
            vectorize_particles=True,
        )

        svi = pyro.infer.SVI(
            model=self.model,
            guide=self.guide,
            optim=self.optimizer,
            loss=elbo,
        )

        self.train()

        self.init_losses = torch.zeros([self.n_cells])
        self.min_losses = torch.zeros([self.n_cells])
        for cell in range(self.n_cells):
            cell_data = data_tensor[data_tensor[:, 3]==cell]
            init_loss = svi.evaluate_loss(cell_data[:, :3], cell)
            self.init_losses[cell] = init_loss
            with pyro.poutine.scale(scale=1./abs(init_loss)):
                min_loss = svi.evaluate_loss(cell_data[:, :3], cell)
                self.min_losses[cell] = min_loss

        wait_epochs = 0
        self.losses = [[] for i in range(self.n_cells)]
        self.memory_stats = [[] for i in range(self.n_cells)]

        # start timer
        if self.device.startswith('cuda'):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            start_time = time.time()

        # optimization loop
        for epoch in range(max_epochs):
            improvements = torch.zeros([self.n_cells])
        
            for cell in range(self.n_cells):
                cell_data = data_tensor[data_tensor[:, 3]==cell]
                with pyro.poutine.scale(scale=1./abs(self.init_losses[cell])):
                    self.zero_grad()
                    loss = svi.step(cell_data[:, :3], cell)
                self.losses[cell].append(loss)
                improvements[cell] = self.min_losses[cell] - loss

                print(
                    'epoch: %s, cell: %s, improvement: %s, patience: %s'
                    %(epoch, cell, round(improvements[cell].item(), 4),
                    early_stopping_patience - wait_epochs)
                )

                if self.device.startswith('cuda'):
                    self.memory_stats[cell].append(torch.cuda.memory_stats())
                torch.cuda.empty_cache()

            # save model
            if save:
                if (epoch % save_every) == 0:
                    self.epochs = epoch
                    if self.device.startswith('cuda'):
                        end.record()
                        torch.cuda.synchronize()
                        self.runtime = start.elapsed_time(end)
                    else:
                        self.runtime = start_time - time.time()

                    self.save_model(
                        path=os.path.join(save_dir, 'epoch_{}/'.format(epoch))
                    )

            # early stopping
            if (improvements >= min_improvement).any() & (epoch > 10):
                wait_epochs = 0
                self.min_losses -= improvements.clamp(min_improvement)

            else:
                wait_epochs += 1

            if wait_epochs > early_stopping_patience:
                self.epochs = epoch
                if self.device.startswith('cuda'):
                    end.record()
                    torch.cuda.synchronize()
                    self.runtime = start.elapsed_time(end)
                else:
                    self.runtime = start_time - time.time()
                if save:
                    self.save_model(path=os.path.join(save_dir, 'final/'))
                break

    def get_factors(
        self,
        n_samples: int=25,
        ) -> torch.tensor:
        """Get the inferred means of the factors.
        
        Args:
            n_samples: number of samples to use for mean estimate

        Returns:
            tensor of shape (n_cells, n_factors, res, res)
        """
        self.eval()
        
        factor_means = []
        with torch.no_grad():
            for cell_id in range(self.n_cells):
                dist = self.factor_list[cell_id](self.grid)
                samples = dist(torch.Size([n_samples])).cpu()
                mean = samples.mean(dim=0).view(
                    self.n_factors, self.grid_res, self.grid_res)
                mean = torch.transpose(mean, -1, -2)
                mask = self.masks[cell_id].cpu().to(dtype=torch.float32)
                mask[mask < 0.5] = np.nan
                mean *= mask.T
                factor_means.append(torch.transpose(mean, -1, -2))

        factor_means = torch.stack(factor_means, dim=0)
        factor_means = torch.nn.Softplus()(factor_means)

        self.train()

        return factor_means

    def get_weights(
        self,
        ) -> torch.tensor:
        """Get the inferred means of the weights.
        
        Returns:
            inferred weights with shape (n_genes, n_factors)
        """
        q_loc = pyro.param('q_loc').detach().cpu().squeeze()

        return torch.nn.Softplus()(q_loc)

    def save_model(
        self,
        path: str,
        ) -> dict:
        """Save the model.

        Args:
            path: directory where everything is saved
        """
        if not os.path.isdir(path):
            os.makedirs(path)

        # factors
        for cell in range(self.n_cells):
            torch.save(
                obj=self.factor_list[cell].state_dict(),
                f=os.path.join(path, 'factors_{}.pt'.format(cell)),
                )
            
        # Pyro parameters
        pyro.get_param_store().save(os.path.join(path, 'pyro_params.save'))

        # model configuration
        config = {
            'n_factors' : self.n_factors,
            'device' : self.device,
            'n_inducing' : self.n_inducing,
            'grid_res' : self.grid_res,
            'factor_smoothness' : self.factor_smoothness,
            'masks_threshold' : self.masks_threshold,
            'init_bin_res' : self.init_bin_res,
        }
        json.dump(
            obj=config,
            fp=open(os.path.join(path, 'config.json'), 'w')
        )

        # latents
        latents = {
            'w' : self.get_weights(),
            'z' : self.get_factors(),
        }
        pickle.dump(
            obj=latents,
            file=open(os.path.join(path, 'latents.pkl'), 'wb')
        )

        # other things
        other = {
            'genes' : self.genes,
            'cells' : self.cells,
            'n_cells' : self.n_cells,
            'n_genes' : self.n_genes,
            'grid' : self.grid.cpu(),
            'masks' : self.masks.cpu(),
            'w_init' : self.w_init.cpu(),
            'q_init' : self.q_init.cpu(),
            'init_losses' : self.init_losses,
            'min_losses' : self.min_losses,
            'losses' : self.losses,
            'memory_stats' : self.memory_stats,
            'runtime' : self.runtime,
        }
        pickle.dump(
            obj=other,
            file=open(os.path.join(path, 'other.pkl'), 'wb')
        )
