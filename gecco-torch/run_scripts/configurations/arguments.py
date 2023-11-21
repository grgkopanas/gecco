import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    # I/O
    parser.add_argument('-d', '--dataset_path', required=False)
    parser.add_argument('--output_dir', required=False, default="logs")
    parser.add_argument('--exp_name', required=False, type=str, default="")

    # Dataset 
    parser.add_argument('--n_points', required=False, type=int, default=None)
    parser.add_argument('--epoch_size', required=False, type=int, default=5_000)
    parser.add_argument('--batch_size', required=False, type=int, default=8)

    # Network
    parser.add_argument('--network_type', required=True, type=str,
                        choices=['transformer', 'multiview_gaussian', 'multiview_voxel'])
    parser.add_argument('--voxel_resolution', required=False, type=int, default=128)

    return parser.parse_args()
