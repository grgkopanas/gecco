import torch
from gecco_torch.diffusion import EDMPrecond, Diffusion, IdleConditioner
from gecco_torch.reparam import GaussianReparam
from gecco_torch.diffusion import Diffusion, LogUniformSchedule, EDMLoss
from gecco_torch.data.shapenet_unc import ShapeNetUncondDataModule
from .common import get_network
import numpy as np

def prepare_data(args):
    dataset_path = (
        #"/data/graphdeco/user/gkopanas/point_diffusion/ShapeNetCore.v2.PC15k/"
        "/data/graphdeco/user/gkopanas/point_diffusion/ShapeNet_HD/"
    )

    return ShapeNetUncondDataModule(
        dataset_path,
        category="airplane",
        n_points=args.n_points,
        epoch_size=args.epoch_size,
        batch_size=args.batch_size,
        num_workers=16,
    )

def prepare_model(args):
    reparam = GaussianReparam(
        mean_xyz=np.array([0.0, 0.01, 0.05]),
        mean_fdc=np.array([0.0, 0.0, 0.0]),
        mean_frest=np.array([0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        mean_opacity=np.array([0.0]),
        mean_scale=np.array([0.0]),
        sigma_xyz=np.array([0.11, 0.04, 0.17]),
        sigma_fdc=np.array([0.0, 0.0, 0.0]),
        sigma_frest=np.array([0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        sigma_opacity=np.array([0.0]),
        sigma_scale=np.array([0.0])
    )
    network = get_network(args)

    return Diffusion(
        backbone=EDMPrecond(
            model=network,
        ),
        conditioner=IdleConditioner(),
        reparam=reparam,
        loss=EDMLoss(
            schedule=LogUniformSchedule(
                max=165.0,
            ),
        ),
    )