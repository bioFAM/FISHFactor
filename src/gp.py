import torch
import gpytorch
import pandas as pd


class SpatialFactors(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points: pd.DataFrame,
        n_factors: int,
        smoothness: float=1.5,
        scale_param: bool=False,
        ) -> None:
        """Spatial factors for a single cell using GPs with Matérn kernel
        
        Args:
            inducing_points: DataFrame with columns 'x', 'y', initial inducing
                point locations
            n_factors: number of spatial factors
            smoothness: smoothness parameter of kernel, can be 0.5, 1.5 or 2.5
            scale_param: if True, use kernel scale parameter
        """
        n_inducing = len(inducing_points)

        variational_distribution = (
            gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points=n_inducing,
                batch_shape=torch.Size([n_factors]),
                )
            )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            model=self,
            inducing_points=torch.tensor(
                data=inducing_points[['x', 'y']].values,
                dtype=torch.float32
                ),
            variational_distribution=variational_distribution,
            )

        super().__init__(variational_strategy=variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean(
            batch_shape=torch.Size([n_factors]),
            )

        lengthscale_prior = gpytorch.priors.NormalPrior(0.5, 0.2)
        kernel = gpytorch.kernels.MaternKernel(
            batch_shape=torch.Size([n_factors]),
            nu=smoothness,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=gpytorch.constraints.Interval(0.2, 0.8),
            )
        kernel.lengthscale = lengthscale_prior.loc

        if scale_param:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                base_kernel=kernel,
                batch_shape=torch.Size([n_factors])
                )
        else:
            self.covar_module = kernel

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean, covar)