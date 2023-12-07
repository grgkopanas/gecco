import numpy as np
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
        full_appearance=args.full_appearance
    )

def prepare_model(args):
    reparam = GaussianReparam(
        mean_xyz=np.array([-3.8525e-03, -5.9943e-02, -6.5675e-03]),
        mean_fdc=np.array([-6.3406e-01, -7.5201e-01, -8.2614e-01]),
        mean_frest=np.array([-1.5836e-03, -2.5277e-03, -2.7809e-03,  2.8718e-02, 3.5960e-02,  3.5978e-02, -2.1406e-03, -2.2680e-04,  6.8243e-04]),
        mean_opacity=np.array([3.7060e+00]),
        mean_scale=np.array([-5.7565e+00]),
        sigma_xyz=np.array([0.2664, 0.2503, 0.2817]),
        sigma_fdc=np.array([0.6271, 0.6636, 0.6787]),
        sigma_frest=np.array([0.0633, 0.0650, 0.0681, 0.0760, 0.0783, 0.0759, 0.0611, 0.0669, 0.0734]),
        sigma_opacity=np.array([5.4635]),
        sigma_scale=np.array([0.8686])
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