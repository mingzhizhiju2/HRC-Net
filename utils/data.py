import os
import numpy as np
import torch
from torch.utils.data import Dataset


class ModelNet40Cls(Dataset):
    def __init__(self, root, split="train", num_points=1024):
        self.root = root
        self.split = split
        self.num_points = num_points
        self._load()

    def _load(self):
        data_path = os.path.join(self.root, "modelnet40_ply_hdf5_2048")
        if self.split == "train":
            filelist = os.path.join(self.root, "train_files.txt")
        else:
            filelist = os.path.join(self.root, "test_files.txt")

        with open(filelist) as f:
            h5_files = [line.strip() for line in f.readlines()]

        pts_list, lbl_list = [], []
        for h5f in h5_files:
            import h5py

            full_path = h5f if os.path.isabs(h5f) else os.path.join(data_path, os.path.basename(h5f))
            with h5py.File(full_path, "r") as f:
                pts_list.append(f["data"][:])
                lbl_list.append(f["label"][:])

        self.data = np.concatenate(pts_list, axis=0).astype(np.float32)
        self.labels = np.concatenate(lbl_list, axis=0).squeeze().astype(np.int64)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        pts = self.data[idx]
        choice = np.random.choice(pts.shape[0], self.num_points, replace=True)
        pts = pts[choice, :3]
        pts = pts - pts.mean(axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(pts, axis=1, keepdims=True), axis=0, keepdims=True)
        pts = pts / (scale + 1e-8)
        return torch.from_numpy(pts).float(), self.labels[idx]


class ShapeNetPartSeg(Dataset):
    def __init__(self, root, split="trainval", num_points=2048):
        self.root = root
        self.split = split
        self.num_points = num_points
        self._load()

    def _load(self):
        filelist = os.path.join(self.root, "train_val_files.txt")
        test_filelist = os.path.join(self.root, "test_files.txt")

        if self.split == "trainval":
            fl = filelist
        else:
            fl = test_filelist

        import h5py

        with open(fl) as f:
            h5_files = [line.strip() for line in f.readlines()]

        pts_list, lbl_list, cat_list = [], [], []
        for h5f in h5_files:
            full_path = h5f if os.path.isabs(h5f) else os.path.join(self.root, os.path.basename(h5f))
            with h5py.File(full_path, "r") as f:
                pts_list.append(f["data"][:])
                lbl_list.append(f["label"][:])
                cat_list.append(f["pid"][:])

        self.data = np.concatenate(pts_list, axis=0).astype(np.float32)
        self.labels = np.concatenate(lbl_list, axis=0).astype(np.int64)
        self.categories = np.concatenate(cat_list, axis=0).squeeze().astype(np.int64)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        pts = self.data[idx]
        lbl = self.labels[idx]
        cat = self.categories[idx]
        choice = np.random.choice(pts.shape[0], self.num_points, replace=True)
        pts = pts[choice, :3]
        lbl = lbl[choice]
        return torch.from_numpy(pts).float(), torch.from_numpy(lbl).long(), torch.tensor(cat).long()
