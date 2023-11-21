import os
import numpy as np
import torch
import lightning.pytorch as pl
from plyfile import PlyData, PlyElement

from gecco_torch.structs import Example
from gecco_torch.data.samplers import ConcatenatedSampler, FixedSampler


class Dataset3DGS:
    def __init__(
        self,
        root: str,
        split: str,
        n_points: int,
        batch_size: int = 48,
    ):
        self.n_points = n_points
        self.path = os.path.join(root, split)
        if not os.path.exists(self.path):
            raise ValueError(f"Path {self.path} does not exist")

        self.plys = []
        for d in os.listdir(self.path):
            self.plys.append(os.path.join(self.path, d, "point_cloud", "iteration_7000", "point_cloud.ply"))
        self.batch_size = batch_size

    def __len__(self):
        return len(self.plys)

    def __getitem__(self, idx):
        plydata = PlyData.read(self.plys[idx])
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        #points = np.load(os.path.join(self.path, self.npys[index]))
        #points = torch.from_numpy(points).to(torch.float32)
        perm = torch.randperm(xyz.shape[0])[: self.n_points]
        selected = xyz[perm]
        return Example(selected, [])


class UncondDataModule_3DGS(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        n_points: int,
        batch_size: int = 48,
        num_workers: int = 8,
        epoch_size: int | None = 10_000,
    ):
        super().__init__()

        self.n_points = n_points
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epoch_size = epoch_size

    def setup(self, stage=None):
        self.train = Dataset3DGS(self.root, "train", self.n_points)
        self.val = Dataset3DGS(self.root, "val", self.n_points)
        self.test = Dataset3DGS(self.root, "test", self.n_points)

    def train_dataloader(self):
        if self.epoch_size is None:
            kw = dict(
                shuffle=True,
                sampler=None,
            )
        else:
            kw = dict(
                shuffle=False,
                sampler=ConcatenatedSampler(
                    self.train, self.epoch_size * self.batch_size, seed=None
                ),
            )

        return torch.utils.data.DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            **kw,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=FixedSampler(self.val, length=None, seed=42),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
