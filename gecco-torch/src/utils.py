import math
import torch
from pvd.gaussian_renderer.render_point_cloud import render_pc_naked
from pvd.datasets.sphere_data_pc import get_cameras_fast
from pvd.model.common import get_features, get_opacity, get_scales, get_xyz

def render_batch(batch_data):
    batch_data = batch_data.float()
    fov = math.pi/3.0
    imgs = []
    for i in range(batch_data.shape[0]):
        #p = torch.tensor([int(bit)*2 - 1.0 for bit in format(i, '08b')[-3:]])
        p = torch.tensor([1.0, 1.0, 1.0])
        cam_w2v, cam_proj, cam_center = get_cameras_fast(torch.zeros(1, 3),
                                                        #torch.tensor(polar2cart(math.pi/4.0, math.pi/4.0, 2.0)),
                                                        p,
                                                        FOV=fov,
                                                        type="perspective")
        rots = torch.zeros((1, batch_data.shape[1], 4)).cuda()
        rots[:, :, 0] = 1

        features = get_features(batch_data[i])
        features = features.view(features.shape[0], 4, 3)
        imgs.append(render_pc_naked(xyz = get_xyz(batch_data[i]),
                                    sh = features,
                                    alphas = get_opacity(batch_data[i]),
                                    scales = get_scales(batch_data[i]),
                                    rots = rots,
                                    world_view = cam_w2v.cuda(),
                                    proj = cam_proj.cuda(),
                                    cam_center = cam_center.cuda(),
                                    H=511, W=511,
                                    FOV=fov,
                                    bg_color=torch.tensor([0.5, 0.0, 0.0]).float().cuda()))
    return torch.concatenate(imgs, dim=0) 
