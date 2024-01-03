from pvd.model.multiview import  MultiView_DiffusionModel2, MultiView_DiffusionModel
import torch
from gecco_torch.models.set_transformer import SetTransformer
from gecco_torch.models.linear_lift import LinearLift
from gecco_torch.models.activation import GaussianActivation
from lightning.pytorch import loggers as pl_loggers
import uuid
import os
import lightning.pytorch as pl
from gecco_torch.models.set_transformer import SetTransformer
from gecco_torch.models.linear_lift import LinearLift
from gecco_torch.models.activation import GaussianActivation
from gecco_torch.data.shapenet_unc import ShapeNetUncondDataModule
from gecco_torch.ema import EMACallback
from gecco_torch.vis import PCVisCallback

class MultiView_Wrapper_Voxel(torch.nn.Module):
    def __init__(self, args):
        super(MultiView_Wrapper_Voxel, self).__init__()
        H, W = args.voxel_resolution, args.voxel_resolution
        self.model = MultiView_DiffusionModel2(device=torch.device(0),
                                               num_views_per_step=args.num_views_per_step,
                                               cnn_all_views=args.cnn_all_views,
                                               cnn_out_mult=args.cnn_out_mult,
                                               H=H, W=W,
                                               cam_conditioning=False,
                                               normalize=True,
                                               act_in_viewspace=None,
                                               stationary_cams=True,
                                               full_appearance=args.full_appearance)

    def forward(self, inputs, timesteps, raw_context=None, post_context = None,  do_cache= False, cache= None):
        return self.model(inputs.permute(0,2,1), timesteps.squeeze()).permute(0,2,1)

class MultiView_Wrapper_GS(torch.nn.Module):
    def __init__(self):
        super(MultiView_Wrapper_GS, self).__init__()
        H, W = 128, 128
        self.model = MultiView_DiffusionModel(device=torch.device(0),
                                               num_points=2048,
                                               num_views_per_step=6,
                                               cam_radius=1.0,
                                               H=H, W=W, FOV=3.14/4.0,
                                               cam_conditioning=False,
                                               normalize=True,
                                               act_in_viewspace=None,
                                               stationary_cams=True)

    def forward(self, inputs, timesteps, raw_context=None, post_context = None,  do_cache= False, cache= None):
        return self.model(inputs.permute(0,2,1), timesteps.squeeze()).permute(0,2,1)

def model_transformer(args):
    if args.full_appearance:
        in_channels = 17
    else:
        in_channels = 3

    feature_dim = 3 * 128
    network = LinearLift(
        input_dim=in_channels,
        inner=SetTransformer(
            n_layers=6,
            num_inducers=64,
            feature_dim=feature_dim,
            t_embed_dim=1,
            num_heads=8,
            activation=GaussianActivation,
        ),
        feature_dim=feature_dim,
    )
    return network

def model_multiview_3GS():
    return MultiView_Wrapper_GS()

def model_multiview_voxel(args):
    return MultiView_Wrapper_Voxel(args)


def get_network(args):
    if args.network_type=="multiview_voxel":
        return model_multiview_voxel(args)
    elif args.network_type=="transformer":
        return model_transformer(args)
    elif args.nework_type=="multiview_gaussian":
        return model_multiview_3GS()
    else:
        print("ERROR: args.network_type not recognized.")
        return

def prepare_trainer(args):

    unique_str = str(uuid.uuid4())
    full_exp_name = args.exp_name + "_" + unique_str[0:10]
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.output_dir, name=full_exp_name)
    #wandb_logger = pl_loggers.WandbLogger(save_dir=args.output_dir, name=full_exp_name, offline=True)
    return pl.Trainer(
        logger=tb_logger,
        default_root_dir=os.path.split(__file__)[0],
        callbacks=[
            EMACallback(decay=0.99),
            pl.callbacks.ModelCheckpoint(),
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                filename="{epoch}-{val_loss:.3f}",
                save_top_k=1,
                mode="min",
            ),
            PCVisCallback(n=8, n_steps=128, point_size=0.01),
        ],
        max_epochs=args.max_epoch,
        #precision="16-mixed",
        precision="32",
        gradient_clip_val=1.0,
        gradient_clip_algorithm="value"
    )  