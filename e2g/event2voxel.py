# -*- coding: utf-8 -*-
# @Time: 2024/11/28
# @File: event2voxel.py
# @Author: fwb
import numpy as np
import torch
from spconv.pytorch.utils import PointToVoxel


def min_max_normalization(arr, mx, mi=0):
    arr = arr.astype(float)
    epsilon = 1e-8
    min_val = np.min(arr, axis=0)
    max_val = np.max(arr, axis=0)
    de = max_val - min_val
    if np.any(de < epsilon):
        if arr.ndim == 1:
            arr[:] = 0.5
        else:
            small_index = np.where(de < epsilon)[0]
            big_index = np.where(de >= epsilon)[0]
            arr[:, small_index] = 0.5
            arr[:, big_index] = (mx - mi) * ((arr[:, big_index] - min_val[big_index]) /
                                             (max_val[big_index] - min_val[big_index])) + mi
        normalize_data = arr
    else:
        normalize_data = (mx - mi) * ((arr - min_val) / (max_val - min_val)) + mi
    return normalize_data


class VoxelGenerator:
    def __init__(self, args, events_dict, device=torch.device("cuda")):
        assert len(args.coord_range) == 3 and len(args.grid_size) == 3, "Wrong coord size!"
        self.args = args
        self.device = device
        self.x_range, self.y_range, self.t_range = np.array(args.coord_range).astype(np.float32)
        self.vx, self.vy, self.vt = np.array(args.grid_size).astype(np.int64)
        events_dict['t'] = min_max_normalization(events_dict['t'], mx=self.t_range)
        self.events_arr = np.hstack((
            events_dict['x'].reshape(-1, 1),
            events_dict['y'].reshape(-1, 1),
            events_dict['t'].reshape(-1, 1),
            events_dict['p'].reshape(-1, 1)
        )).astype(np.float32)

    def to_voxel(self):
        voxel_generator = PointToVoxel(
            vsize_xyz=[self.vx, self.vy, self.vt],
            coors_range_xyz=[0, 0, 0,
                             self.x_range,
                             self.y_range,
                             self.t_range
                             ],
            num_point_features=self.events_arr.shape[1],
            max_num_voxels=self.args.max_num_voxels,
            max_num_points_per_voxel=self.args.max_num_points_per_voxel,
            device=self.device
        )
        events_tensor = torch.from_numpy(self.events_arr).to(device=self.device)
        voxels, voxel_coords, num_points_per_voxel = voxel_generator(events_tensor)
        # Convert tyx to xyt.
        voxel_coords[:, [0, 1, 2]] = voxel_coords[:, [2, 1, 0]]
        # Filter all null voxels.
        if self.args.is_filter:
            mask = ~torch.all(voxels.view(voxels.size(0), -1) == 0, dim=1)
            voxels = voxels[mask]
            voxel_coords = voxel_coords[mask]
        # Select the voxel with the top Ne events.
        if num_points_per_voxel.shape[0] < self.args.Nv:
            sel_voxels = voxels  # x, y, t, p
            voxel_coords = voxel_coords  # x, y, t
        else:
            _, valid_index = torch.topk(num_points_per_voxel, k=self.args.Nv)
            sel_voxels = voxels[valid_index]
            voxel_coords = voxel_coords[valid_index]
        # Normalized the time of the event within the voxel 0 to 1.
        in_voxel_normal_t = torch.fmod(sel_voxels[:, :, 2], self.vt) / self.vt
        # Calculate each voxel feature.
        in_voxel_xy = sel_voxels[:, :, 0:2]
        in_voxel_p = sel_voxels[:, :, 3]
        voxel_feats = torch.zeros(sel_voxels.shape[0],
                                  3,
                                  int(self.vx * self.vy)).to(self.device)
        # Mapping intra voxel events xy to vx * xy coordinates.
        unique_coords = (torch.fmod(in_voxel_xy[:, :, 0], self.vx) +
                         self.vx * torch.fmod(in_voxel_xy[:, :, 1], self.vy)).to(torch.int64)
        for feat_dim in range(1, 4):
            match feat_dim:
                case 1:
                    voxel_feats[:, feat_dim - 1, :].scatter_add_(1,
                                                                 index=unique_coords,
                                                                 src=in_voxel_normal_t * in_voxel_p)
                case 2:
                    double_t = in_voxel_normal_t * 2
                    bid_t = 1 - torch.abs(1 - double_t)
                    voxel_feats[:, feat_dim - 1, :].scatter_add_(1,
                                                                 index=unique_coords,
                                                                 src=bid_t * in_voxel_p)
                case 3:
                    reverse_t = torch.abs(in_voxel_normal_t - 1)
                    voxel_feats[:, feat_dim - 1, :].scatter_add_(1,
                                                                 index=unique_coords,
                                                                 src=reverse_t * in_voxel_p)
                case _:
                    print(f"Number of feature channels {feat_dim} does not exist!")
        voxel_feats = voxel_feats.view(-1, 3, self.vx, self.vy)
        # Time ascending order.
        _, sorted_index = torch.sort(voxel_coords[:, -1])
        voxel_feats = voxel_feats[sorted_index]
        voxel_coords = voxel_coords[sorted_index]
        return voxel_feats.cpu().numpy().copy(), voxel_coords.cpu().numpy().copy()
