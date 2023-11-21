import torch
from gecco_torch.diffusion import EDMPrecond, Diffusion, IdleConditioner
from gecco_torch.reparam import GaussianReparam
from gecco_torch.diffusion import Diffusion, LogUniformSchedule, EDMLoss
from gecco_torch.data.gaussians_unc import UncondDataModule_3DGS
from .common import get_network

def prepare_data(args):
    dataset_path = (
        "/data/graphdeco/user/gkopanas/point_diffusion/datasets/train_3DGS_scape_cycles_fixed25k_split/"
    )

    return UncondDataModule_3DGS(
        dataset_path,
        n_points=args.n_points,
        epoch_size=args.epoch_size,
        batch_size=args.batch_size,
        num_workers=16,
    )

def prepare_model(args):
    reparam = GaussianReparam(
        mean=torch.tensor([0.0, 0.0, 0.0]),
        sigma=torch.tensor([0.2536, 0.2422, 0.2902]),
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