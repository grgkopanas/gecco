from configurations.shapenet_airplane import prepare_data, prepare_model
from configurations.common import prepare_trainer
from configurations.arguments import get_args
import torch
from utils import save_ply_files
from pathlib import Path
import os
from tqdm import tqdm

def gen_shapes(args):
    model = prepare_model(args)
    data = prepare_data(args)
    trainer = prepare_trainer(args)


    checkpoint_state_dict = torch.load(args.model_path, map_location='cpu')
    model_state_dict = checkpoint_state_dict['ema_state_dict']
    model.load_state_dict(model_state_dict)

    model = model.to("cuda").eval()
    
    torch.manual_seed(42)
    TOTAL_PCS = 256
    BATCH_SIZE = 8
    for b in tqdm(range(0, TOTAL_PCS, BATCH_SIZE)):
        samples = model.sample_stochastic(shape=(BATCH_SIZE, 2048, 3),
                                          context=None,
                                          sigma_max=model.sigma_max,
                                          num_steps=64)
        save_ply_files(samples.detach().cpu().numpy(), 
                       os.path.join(os.path.dirname(args.model_path), Path(args.model_path).stem), 
                       offset_id=b)

if __name__ == "__main__":
    args = get_args()
    args.model_path = "/data/graphdeco/user/gkopanas/point_diffusion/gecco/gecco-torch/run_scripts/logs/shapenet_scratch/test00003__b0dbdd4d-1/version_0/checkpoints/epoch=46-val_loss=81.547.ckpt"
    gen_shapes(args)