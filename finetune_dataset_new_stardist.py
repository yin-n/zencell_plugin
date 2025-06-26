#!/usr/bin/env python

import torch
import pandas as pd
from typing import Tuple
import os
import zarr
import numpy as np
import tifffile
import torch.nn.functional as F

import dask.array as da

from stardist import edt_prob, star_dist

class BrainCellDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_root: str,
        split: str,
        img_size: Tuple[int, int, int],
        ctx_size: Tuple[int, int, int],
        alignment: Tuple[int, int, int],
    ) -> None:
        assert split in ["train", "val"]
        self.data_root = data_root
        self.split = split
        self.img_size = img_size
        self.ctx_size = ctx_size
        self.alignment = alignment

        df = pd.read_csv(os.path.join(data_root, f"{split}.csv"), dtype={"ID": str})
        self.ids = df["ID"].tolist()
        self.zarr_names = [x for x in df["source"].tolist()]
        self.corners = [
            tuple(int(y.strip()) for y in x.lstrip(" [").rstrip(" ]").split(","))
            for x in df["corner"].tolist()
        ]
        self.crop_sizes = [
            tuple(int(y.strip()) for y in x.lstrip(" [").rstrip(" ]").split(","))
            for x in df["crop_size"].tolist()
        ]
        self.ref_channels = df["ref_channel"].tolist()
        self.channels = df["channel"].tolist()
        self.plane_positions = [round(x) for x in df["plane_position"].tolist()]
        
        self.zarr_fp = {}
        for item in set(self.zarr_names):
            if 'fused.zarr' in item:
                self.zarr_fp[item] = da.from_zarr(zarr.open(item, mode="r")['0'])
            else:
                self.zarr_fp[item] = zarr.open(store=zarr.N5Store(item), mode="r")
        
        self.label_roots = df["label_root"].tolist()
        
        self.n_rays=96
        self.b = slice(None),slice(None)
        self.grid=tuple((1,1))
        self.ss_grid = (slice(None),) + tuple(slice(0, None, g) for g in (1,1))

    def __len__(self) -> int:
        return len(self.ids)

    def _load_and_merge_masks(self, img_id: str, label_root: str) -> np.array:
        cell_id_offset = 0
        merged_arr = None
        for i in [1, 2, 3]:
            part_arr = tifffile.imread(os.path.join(label_root, self.split, f"{img_id}_mask{i}.tif"))
            if merged_arr is None:
                merged_arr = part_arr
            else:
                assert merged_arr.shape == part_arr.shape
                part_arr[part_arr != 0] += cell_id_offset
                merged_arr = np.maximum(merged_arr, part_arr)
            cell_id_offset = merged_arr.max()
        return merged_arr

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.zarr_fp is None:
            self.zarr_fp = {
                key: zarr.open(store=zarr.N5Store(value), mode="r")
                for key, value in name_to_zarr_path.items()
            }
                

        zarr_name = self.zarr_names[idx]
        ref_channel = self.ref_channels[idx]
        channel = self.channels[idx]

        corner = self.corners[idx]
        plane_position = int(self.plane_positions[idx])
        crop_size = self.crop_sizes[idx]
        img_id = self.ids[idx]
        label_root = self.label_roots[idx]

        label_vol_st = (corner[0] + plane_position, corner[1], corner[2])
        label_vol_ed = (corner[0] + plane_position + 1, corner[1] + crop_size[1], corner[2] + crop_size[2])
        if os.path.isfile(os.path.join(label_root, self.split, f"{img_id}_mask.tif")):
            label_vol = [tifffile.imread(os.path.join(label_root, self.split, f"{img_id}_mask.tif"))]
        else:
            label_vol = [self._load_and_merge_masks(img_id, label_root)]

        label_flow = [np.transpose(star_dist(lbl,self.n_rays,mode='cpp', grid=self.grid), (2,0,1)) for lbl in label_vol]
        label_bin = [edt_prob(lbl[self.b][self.ss_grid[1:3]]) for lbl in label_vol]
        label_vol = np.stack(label_vol, axis=0)
        label_flow = np.stack(label_flow, axis=1)
        label_bin = np.stack(label_bin, axis=0)
        assert label_vol.shape == tuple(y - x for x, y in zip(label_vol_st, label_vol_ed))
        assert label_flow.shape == (self.n_rays, *label_vol.shape), label_flow.shape
        assert label_bin.shape == label_vol.shape
        label_mask = np.ones_like(label_vol, dtype=bool)

        slice_arr = []
        pad_arr = []
        new_label_vol_st, new_label_vol_ed = [], []
        for dim in range(3):
            if label_vol.shape[dim] <= self.img_size[dim]:
                slice_arr.append(slice(0, label_vol.shape[dim], 1))
                tot_pad = self.img_size[dim] - label_vol.shape[dim]
                pad_arr.append((tot_pad // 2, (tot_pad + 1) // 2))
                new_label_vol_st.append(label_vol_st[dim] - tot_pad // 2)
                new_label_vol_ed.append(label_vol_ed[dim] + (tot_pad + 1) // 2)
            else:
                tot_slice = label_vol.shape[dim] - self.img_size[dim]
                slice_arr.append(slice(tot_slice // 2, label_vol.shape[dim] - (tot_slice + 1) // 2, 1))
                pad_arr.append((0, 0))
                new_label_vol_st.append(label_vol_st[dim] + tot_slice // 2)
                new_label_vol_ed.append(label_vol_ed[dim] - (tot_slice + 1) // 2)
        label_vol_st = tuple(new_label_vol_st)
        label_vol_ed = tuple(new_label_vol_ed)
        label_vol, label_mask = label_vol[tuple(slice_arr)], label_mask[tuple(slice_arr)]
        label_vol, label_mask = np.pad(label_vol, pad_arr), np.pad(label_mask, pad_arr)
        label_flow = label_flow[(slice(None),) + tuple(slice_arr)]
        label_flow = np.pad(label_flow, [(0, 0)] + pad_arr)
        label_bin = label_bin[tuple(slice_arr)]
        label_bin = np.pad(label_bin, pad_arr)

        assert label_vol.shape == label_mask.shape and label_vol.shape == tuple(y - x for x, y in zip(label_vol_st, label_vol_ed))
        img_slice_arr, img_pad_arr = [], []

        if 'fused.zarr' in zarr_name:
            zarr_shape = self.zarr_fp[zarr_name][3].shape
        else:
            zarr_shape = self.zarr_fp[zarr_name]["setup3"]["timepoint0"]["s0"].shape

        assert len(zarr_shape) == 3
        for dim in range(3):
            roi_offs = (self.ctx_size[dim] - self.img_size[dim]) // 2 // self.alignment[dim] * self.alignment[dim]
            input_vol_st = label_vol_st[dim] - roi_offs
            input_vol_ed = input_vol_st + self.ctx_size[dim]
            img_slice_arr.append(slice(max(input_vol_st, 0), min(input_vol_ed, zarr_shape[dim]), 1))
            img_pad_arr.append((-min(input_vol_st, 0), max(input_vol_ed - zarr_shape[dim], 0)))

        input_vol = []
        for chn in [ref_channel, channel]:
            if 'fused.zarr' in zarr_name:
                arr_slice = self.zarr_fp[zarr_name][chn][tuple(img_slice_arr)].compute()
            else:
                arr_slice = self.zarr_fp[zarr_name][f"setup{chn}"]["timepoint0"]["s0"][tuple(img_slice_arr)]
            input_vol.append(arr_slice)
        input_vol = np.stack(input_vol, axis=0)


        input_vol = input_vol - input_vol.mean(axis=(1, 2, 3), keepdims=True)
        input_vol = input_vol / (input_vol.std(axis=(1, 2, 3), keepdims=True) + 1e-6)
        input_vol = np.pad(input_vol, [(0, 0)] + img_pad_arr)

        #label_bin = label_vol >= 1

        if self.split == "train":
            return tuple(torch.tensor(x, dtype=torch.float) for x in (input_vol, label_mask, label_bin, label_flow))
        else:
            return torch.tensor(input_vol, dtype=torch.float), torch.tensor(label_vol)


if __name__ == "__main__":
    dataset = BrainCellDataset(
        data_root="/mnt/aperto/tatz_brain_data/anno_241011",
        split="train",
        img_size=(10, 256, 256),
        ctx_size=(10, 1280, 1280),
        alignment=(2, 64, 64),
    )
    for item in dataset:
        for x in item:
            print(x.size(), end=" ")
        print()

