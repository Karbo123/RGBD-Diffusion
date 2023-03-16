""" ScanNet
    NOTE this dataset loads all frames from a scene at one time
"""

import re
import cv2
import numpy as np
import os.path as osp
from glob import glob
from numpy.lib.recfunctions import unstructured_to_structured

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from einops import rearrange



class ScanNetScene(Dataset):
    def __init__(self,
                 path, 
                 num_views_sample=4, # num of views per scene for UNet compute; set to None to load all
                 max_len_seq=64, # maximum length of sequence; set to None to load all
                 img_size=128,
                 normalize=True,
                 inter_mode="nearest", # sharper is better ??
                 subset_indices=None,
                 reverse_frames=False, # also include reversed time order
                ):
        super().__init__()
        # get file path
        color_list = sorted(glob(osp.join(path, "*", "color", "*.jpg"), recursive=True))
        depth_list = sorted(glob(osp.join(path, "*", "depth", "*.png"), recursive=True))
        pose_list  = sorted(glob(osp.join(path, "*", "pose",  "*.txt"), recursive=True))
        intr_list  = glob(osp.join(path, "*", "intrinsic", "intrinsic_depth.txt"), recursive=True)
        file_path  = np.rec.fromarrays(
                        [color_list, depth_list, pose_list], 
                        names=("color", "depth", "pose"),
                    )
        # sort all views
        to_int = lambda x: [int(i) for i in x]
        number_id = np.asarray([to_int(re.findall("\d+",  osp.relpath(p, path))) \
                                for p in file_path.color])
        sorted_id = unstructured_to_structured(number_id,
                        np.dtype([("col0", int), ("col1", int), ("col2", int)])
                    ).argsort(axis=0, order=("col0", "col1", "col2"))
        file_path = file_path[sorted_id]
        number_id = number_id[sorted_id]
        # sub-sample dataset
        if isinstance(subset_indices, (tuple, list)):
            file_path = file_path[np.asarray(subset_indices)]
            number_id = number_id[np.asarray(subset_indices)]
        elif isinstance(subset_indices, float):
            assert 0.0 < subset_indices <= 1.0
            subset_indices = round(len(file_path) * subset_indices)
            if subset_indices > len(file_path) / 2: # training set has more data
                file_path = file_path[:subset_indices] # train
                number_id = number_id[:subset_indices]
            else:
                file_path = file_path[-subset_indices:] # test
                number_id = number_id[-subset_indices:]
        # identify camera intrinsic
        cache_intr = np.stack([np.loadtxt(p)[:3, :3] for p in intr_list], axis=0) # (?, 3, 3)
        cache_intr_keys = ["|".join(re.findall("\d+",  osp.relpath(p, path))) for p in intr_list]
        cache_intr_colle = list()
        for triple in number_id:
            triple_key = f"{triple[0]:04}|{triple[1]:02}"
            ind = cache_intr_keys.index(triple_key)
            cache_intr_colle.append(cache_intr[ind])
        cache_intr = np.stack(cache_intr_colle, axis=0) # (?, 3, 3)
        cache_intr *= img_size / 480
        cache_intr[:, 0, 2] -= (img_size/2) * 0.333333
        cache_intr[:, 2, 2]  = 1.0
        cache_intr = torch.from_numpy(cache_intr).inverse().float() # to c2w
        # find how many scenes in the dataset
        scene_split_index = []
        for index, nid in enumerate(number_id):
            scene_key = f"{nid[0]}|{nid[1]}"
            if scene_key not in [x[0] for x in scene_split_index]:
                scene_split_index.append((scene_key, index))
        # gather all frames for each scene
        info_chunk = list()
        num_scenes = len(scene_split_index)
        for index_scene in range(num_scenes):
            ind_start = scene_split_index[index_scene    ][1]
            ind_end   = scene_split_index[index_scene + 1][1] \
                            if index_scene < num_scenes - 1 else len(number_id)
            info_chunk.append(list(range(ind_start, ind_end)))
        # save
        self.file_path = file_path
        self.info_chunk = info_chunk
        self.cache_intr = cache_intr
        #
        self.img_size = img_size
        self.normalize = normalize
        self.inter_mode = inter_mode
        self.max_len_seq = max_len_seq
        self.reverse_frames = reverse_frames
        self.num_views_sample = num_views_sample
        # 
        self.normalize_mean_std = torch.tensor([
            (-0.24595006, 0.54566), # R
            (-0.13202806, 0.55846), # G
            (-0.02775778, 0.57430), # B
            ( 1.72314970, 0.99023), # D
        ])
    

    def __len__(self):
        return len(self.info_chunk)

    
    def __getitem__(self, ind_chunk):
        views_indices = self.info_chunk[ind_chunk]
        if self.reverse_frames and torch.rand(1) < 0.5:
            views_indices = list(reversed(views_indices)) # reversed time order
        if isinstance(self.max_len_seq, int) and len(views_indices) > self.max_len_seq:
            st = torch.randint(0, len(views_indices) - self.max_len_seq + 1, size=(1,)).item()
            views_indices = views_indices[st:(st + self.max_len_seq)] # randomly crop
        # 
        num_views = len(views_indices)
        mask = torch.zeros(num_views, dtype=torch.bool)
        if self.num_views_sample is None:
            mask[:] = True
        else:
            mask_ind = torch.multinomial(
                torch.ones(num_views), 
                num_samples=self.num_views_sample, 
                replacement=False,
            )
            mask[mask_ind] = True
            # 
            num_views = mask_ind.max().add(1).item()
            mask = mask[:num_views]
            views_indices = views_indices[:num_views]
        # 
        rgbd, pose, intr = [], [], []
        for ind_view in views_indices:
            all_path = self.file_path[ind_view]
            # 
            img_color = cv2.imread(all_path.color, cv2.IMREAD_UNCHANGED) \
                           .astype(np.float32)[..., [2, 1, 0]] / 127.5 - 1.0 # -1~1
            img_depth = cv2.imread(all_path.depth, cv2.IMREAD_UNCHANGED) \
                           .astype(np.float32) / 1000.0 # in meters
            mat_pose  = np.loadtxt(all_path.pose, dtype=np.float32)[:3, :] # c2w
            mat_intr  = self.cache_intr[ind_view] # c2w
            # 
            img_color = torch.from_numpy(img_color)
            img_depth = torch.from_numpy(img_depth)
            mat_pose  = torch.from_numpy(mat_pose )
            # 
            img_rgbd = torch.cat([
                rearrange(img_color, "H W C -> C H W"),
                img_depth[None, ...],
            ], dim=0)
            # 
            if self.normalize:
                mean, std = self.normalize_mean_std.unbind(dim=1)
                img_rgbd = (img_rgbd - mean[:, None, None]) / std[:, None, None]
            rgbd.append(img_rgbd)
            pose.append(mat_pose)
            intr.append(mat_intr)
        # 
        rgbd = torch.stack(rgbd, dim=0) # (num_views, 3+1, H, W)
        pose = torch.stack(pose, dim=0) # (num_views, 3, 4)
        intr = torch.stack(intr, dim=0) # (num_views, 3, 3)
        # 
        rgbd = F.interpolate(rgbd[:, :, :, 30:210], # crop central region
            size=(self.img_size, self.img_size), # then resize
            mode=self.inter_mode,
            **(dict(align_corners=False) if self.inter_mode == "bilinear" else dict()),
        )
        #
        return dict(rgbd=rgbd, pose=pose, intr=intr, mask=mask,
                    views_indices=torch.tensor(views_indices))



def collate_fn(data):
    out = {k: torch.cat([d[k] for d in data], dim=0) for k in data[0].keys()}
    # views for UNet
    view_index = out["mask"].nonzero().squeeze(1)
    num_views_lst = [len(d["mask"]) for d in data]
    index_range = list()
    start = 0
    for i, x in enumerate(num_views_lst):
        vinds = view_index[(view_index >= start) & (view_index < start + x)].tolist()
        for vind in vinds:
            index_range.append([i, start, vind + 1]) # sceneid, start, end
        start += x
    out["index_range"] = torch.tensor(index_range)
    # 
    return out

