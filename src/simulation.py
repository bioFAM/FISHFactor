from typing import Sequence
import os
import torch
import imageio
from scipy.ndimage import gaussian_filter


def simulate_pp(
    n_features: int,
    factor_dir: str,
    weight_scales_raw: Sequence[float],
    weight_sparsities: Sequence[float],
    sigma: float=2.,
    intensity_scale: float=100.,
    ) -> dict:
    """Simulate a dataset from given factors with random non-negative weights
        and a Poisson process likelihood.

    :param n_features: number of features
    :param factor_dir: directory where factor .png files are stored
    :param weight_scales_raw: weight scale parameters per factor
        (before exponentiation)
    :param weight_sparsities: weight sparsity Bernoulli parameters per factor
    :param sigma: smoothing parameter in gaussian filter applied to factors
    :param intensity_scale: multiplicative factor of Poisson process intensity
    """
    weight_scales_raw = torch.tensor(weight_scales_raw).view(1, -1)
    weight_sparsities = torch.tensor(weight_sparsities).view(1, -1)
    n_latents = len(os.listdir(factor_dir))

    # factors
    z = []
    for k in range(n_latents):
        image = imageio.imread(
            os.path.join(factor_dir, os.listdir(factor_dir)[k])
        )
        image = gaussian_filter(image / 255, sigma)
        image = (image - image.min()) / (image.max() - image.min())
        z.append(torch.tensor(image, dtype=torch.float32))
    z = torch.stack(z, dim=0)

    res = z.shape[1]

    # sparsity
    s = torch.distributions.Bernoulli(
        probs=weight_sparsities.repeat(n_features, 1)
    ).sample()

    # weights
    w = torch.distributions.Normal(
        loc=torch.zeros([n_features, n_latents]),
        scale=weight_scales_raw.repeat(n_features, 1),
    ).sample().exp()

    w *= s

    # avoid having too little intensity for one feature (otherwise requirement
    # in thinning that there is at least 1 point per feature could result in an
    # infinite loop)
    for d in range(n_features):
        while w[d].sum() < 0.5:
            w[d] = torch.distributions.Normal(
                loc=torch.zeros([n_latents]),
                scale=weight_scales_raw,
            ).sample().exp()

    # Poisson process intensity
    intensity = torch.matmul(w, z.view(n_latents, -1))
    intensity *= intensity_scale

    z = z.view(n_latents, res, res)
    intensity = intensity.view(n_features, res, res)

    # simulation by thinning
    data = [] 
    for d in range(n_features):
        while True:
            # homogeneous Poisson process
            max_intensity = intensity[d].max()
            num_points = int(torch.poisson(max_intensity).item())
            points_x = torch.rand([num_points])
            points_y = torch.rand([num_points])

            # thinning
            thinning_probs = (1 - intensity[d] / max_intensity)
            thinning_probs_per_point = thinning_probs[
                (points_y * res).long(),
                (points_x * res).long()
            ]
            rand_probs = torch.rand_like(thinning_probs_per_point)
            keep_points = rand_probs >= thinning_probs_per_point

            feature_data = torch.cat(
                tensors=[
                    points_x[keep_points].view(-1, 1),
                    points_y[keep_points].view(-1, 1),
                    torch.full_like(points_x[keep_points], d).view(-1, 1),
                ],
                dim=1,
            )

            # accept feature data if at least one point was generated
            if keep_points.sum() >= 1:
                break
    
        data.append(feature_data)
    data = torch.cat(data, dim=0) 

    return {
        'n_features' : n_features,
        'factor_dir' : factor_dir,
        'n_latents' : n_latents,
        'weight_scales_raw' : weight_scales_raw,
        'weight_sparsities' : weight_sparsities,
        'sigma' : sigma,
        'intensity_scale' : intensity_scale,
        'z' : z,
        's' : s,
        'w' : w,
        'intensity' : intensity,
        'data' : data,
        'res' : res,
        }