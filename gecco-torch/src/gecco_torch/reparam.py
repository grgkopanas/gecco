"""
Definition of reparameterization schemes. Each reparameterization scheme
implements a `data_to_diffusion` and `diffusion_to_data` method, which
convert the data between "viewable" representation (simple xyz) and the
"diffusion" representation (possibly normalized, logarithmically spaced, etc).
"""
import torch
from torch import Tensor
from kornia.geometry.camera.perspective import project_points, unproject_points

from gecco_torch.structs import Context3d
from pvd.model.common import construct_sample

class GaussianReparam(torch.nn.Module):
    """
    Maps data to diffusion space by normalizing it with the mean and standard deviation.
    """

    def __init__(self, mean_xyz, mean_fdc, mean_frest, mean_opacity, mean_scale,
                       sigma_xyz, sigma_fdc, sigma_frest, sigma_opacity, sigma_scale):
        super().__init__()
        self.register_buffer("mean", torch.tensor(construct_sample(mean_xyz, mean_fdc, mean_frest, mean_opacity, mean_scale)).float())
        self.register_buffer("sigma", torch.tensor(construct_sample(sigma_xyz, sigma_fdc, sigma_frest, sigma_opacity, sigma_scale)).float())

    def data_to_diffusion(self, data: Tensor, ctx: Context3d) -> Tensor:
        del ctx
        mean = self.mean[:data.shape[-1]]
        sigma = self.sigma[:data.shape[-1]]
        return (data - mean) / sigma

    def diffusion_to_data(self, diff: Tensor, ctx: Context3d) -> Tensor:
        del ctx
        mean = self.mean[:diff.shape[-1]]
        sigma = self.sigma[:diff.shape[-1]]
        return diff * sigma + mean

    def extra_repr(self) -> str:
        return f"mean={self.mean.flatten().tolist()}, sigma={self.sigma.flatten().tolist()}"
