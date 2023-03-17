
import torch
import torch.nn as nn
import torch.nn.functional as F


CHUNK_SIZE = 8 # prev + curr
IMG_SIZE = 128
# to be accurate, because scenes in testset cannot appear in trainset, otherwise some views of test scenes are seen
TRAIN_SPLIT_RATIO = 0.8016702610930115

# # # # # # # # # # # # 
FP16_MODE = True
LR_INIT = 1e-4
LR_FINAL = 1e-6
BATCH_SIZE_PER_GPU = 40
SEED = 3407
training = dict(
    epoch_end=300,
)
routine=dict(
    print_every=10,               # per iter
    checkpoint_latest_every=1000, # per iter
    checkpoint_every=8000,        # per iter
    validate_every=8000,          # per iter 
    visualize_every=-1,           # per iter
)
save=dict(
    model_selection_metric="loss",
    model_selection_mode="minimize",
    backup_lst=[
        "dataset/ScanNet",
    ],
)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from recon.utils import dist_info

rank, num_rank, device = dist_info()
BATCH_SIZE = BATCH_SIZE_PER_GPU * num_rank # total batch size (sum of all processes)
torch.manual_seed(SEED)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import os.path as osp
from functools import partial
from dataset import ScanNet
from recon.utils import dist_samplers, kwargs_shuffle_sampler


dataset_create_fn = partial(ScanNet,
                            osp.abspath("./data_file/ScanNetV2"),
                            chunk_size=CHUNK_SIZE, img_size=IMG_SIZE, 
                            normalize=True, inter_mode="nearest")
dataset_train = dataset_create_fn(subset_indices = TRAIN_SPLIT_RATIO)
dataset_test  = dataset_create_fn(subset_indices = 1.0 - TRAIN_SPLIT_RATIO)
# 
samplers = dist_samplers(dataset_train, dataset_test)
dataloaders = dict(train = torch.utils.data.DataLoader(
                                dataset_train, **kwargs_shuffle_sampler(samplers, "train"), 
                                batch_size=BATCH_SIZE_PER_GPU, num_workers=4,
                            ),
                   val   = torch.utils.data.DataLoader(
                                dataset_test, **kwargs_shuffle_sampler(samplers, "val"  ), 
                                batch_size=BATCH_SIZE_PER_GPU, num_workers=2,
                            ),
                )

mean_std = dataset_train.normalize_mean_std.to(device=device)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import trimesh
from einops import rearrange
import nvdiffrast.torch as dr


class Render(object):
    def __init__(self, extend_eps=0.04, depth_min=0.1, edge_max=0.1):
        super().__init__()
        self.extend_eps = extend_eps
        self.depth_min  = depth_min # in meters
        self.edge_max   = edge_max  # in meters
        self.ctx_render = dr.RasterizeCudaContext() # require CUDA
    

    @staticmethod
    def gmm(grouped_left, right, ind_group):
        """ perform grouped matrix multiplication
        """
        result = list()
        for ind, (ind_start, cnt) in enumerate(ind_group):
            mat_left = grouped_left[ind_start : (ind_start + cnt)]
            mat_right = right[ind]
            mat_out = mat_left @ mat_right
            result.append(mat_out)
        result = torch.cat(result, dim=0)
        return result


    def unproject(self, attr_d, c2w_int, c2w_ext, flatten=True):
        """ unproject a 2d image into 3d space
        Args:
            attr_d.shape == (H, W, num_attr + 1) or (H, W)
            c2w_int.shape == (3, 3)
            c2w_ext.shape == (3, 4)
            flatten: whether to flatten (H, W) to (H*W, )
        """
        device = attr_d.device
        H, W = attr_d.shape[:2]
        if attr_d.ndim == 2: attr_d = attr_d[..., None]
        num_attr = attr_d.size(2) - 1
        def make_linspace(num):
            lin = torch.linspace(0.5, num - 0.5, num, device=device)
            lin[-1] += self.extend_eps # extend upper boundary location
            return lin
        x, y = torch.meshgrid(make_linspace(W), 
                              make_linspace(H),
                              indexing="xy",
                            ) # each == (H, W)
        z = attr_d[..., -1] # (H, W)
        a = None
        if num_attr > 0:
            a = attr_d[..., :-1] # (H, W, num_attr)
            a[ -1, :-1] += (a[ -1, :-1] - a[ -2, :-1]) * self.extend_eps
            a[:-1,  -1] += (a[:-1,  -1] - a[:-1,  -2]) * self.extend_eps
            a[ -1,  -1] += (a[ -1,  -1] - a[ -2,  -2]) * self.extend_eps
        # 
        v = torch.stack([x * z, y * z, z], dim=2) # (H, W, 3)
        v = torch.einsum(f"ij,...j->...i", c2w_int, v) # (H, W, 3)
        v = torch.cat([v, v.new_ones([H, W, 1])], dim=2) # (H, W, 4)
        v = torch.einsum(f"ij,...j->...i", c2w_ext, v) # (H, W, 3)
        if flatten:
            v = rearrange(v, "H W C -> (H W) C")
            if num_attr > 0:
                a = rearrange(a, "H W C -> (H W) C")
        return v, a


    @staticmethod
    def mesh_structure(H, W, device="cpu"):
        x, y = torch.meshgrid(torch.arange(W, device=device), 
                              torch.arange(H, device=device), 
                              indexing="xy",
                            ) # each == (H, W)
        pts = y[:-1, :-1].mul(W) + x[:-1, :-1]
        lower = torch.stack([pts, pts + W, pts + W + 1], dim=2).reshape(-1, 3)
        upper = torch.stack([pts, pts + W + 1, pts + 1], dim=2).reshape(-1, 3)
        faces = torch.cat([lower, upper], dim=0) # (num_faces, 3)
        return faces


    def meshing(self, attr_d, c2w_int, c2w_ext, trunc=True):
        """ meshing a 2d image into a 3d mesh
        Args:
            attr_d.shape == (H, W, num_attr + 1) or (H, W)
            c2w_int.shape == (3, 3)
            c2w_ext.shape == (3, 4)
        """
        v, a = self.unproject(attr_d, c2w_int, c2w_ext, flatten=True)
        f = self.mesh_structure(*attr_d.shape[:2], device=attr_d.device)
        if trunc:
            # remove those with small depth values
            d = (attr_d if attr_d.ndim == 2 else attr_d[..., -1]).reshape(-1) # (H*W, )
            msk0 = (d[f] > self.depth_min).all(dim=1) # (num_faces, )
            # remove those with too long edges
            edge_len = torch.stack([
                (v[f[:, 0]] - v[f[:, 1]]).norm(dim=1),
                (v[f[:, 1]] - v[f[:, 2]]).norm(dim=1),
                (v[f[:, 2]] - v[f[:, 0]]).norm(dim=1),
            ], dim=1) # (num_faces, 3)
            msk1 = (edge_len < self.edge_max).all(dim=1) # (num_faces, )
            # remove faces
            f = f[msk0 & msk1]
            # remove verts
            ind_v, f = f.reshape(-1).unique(return_inverse=True)
            v = v[ind_v]
            a = a[ind_v]
            f = f.reshape(-1, 3)
        return v, f, a # vertices, faces, attributes


    def meshing_many(self, attr_d, c2w_int, c2w_ext, trunc=True):
        """ meshing many images into many meshes, stacking into a batch ready for rendering
        Args:
            attr_d.shape == (B, num_attr + 1, H, W)
            c2w_int.shape == (B, 3, 3)
            c2w_ext.shape == (B, 3, 4)
        Returns:
            mesh (tuple): 
                containing (vertices, faces, attrs, ind_group)
        """
        has_attr = attr_d.size(1) > 1
        vertices, faces, attrs = [], [], []
        offset = 0
        for attrd, camint, camext in zip(attr_d, c2w_int, c2w_ext):
            attrd = rearrange(attrd, "C H W -> H W C") # (H, W, num_attr + 1)
            v, f, a = self.meshing(attrd, camint, camext, trunc=trunc)
            vertices.append(v)
            faces.append(f + offset)
            if has_attr: attrs.append(a)
            offset += len(v)
        num_faces = torch.tensor([len(f) for f in faces]) # on CPU
        num_verts = torch.tensor([len(v) for v in vertices]) # on CPU
        vertices = torch.cat(vertices, dim=0)
        faces    = torch.cat(faces,    dim=0)
        attrs    = torch.cat(attrs,    dim=0) if has_attr else None
        ind_group = torch.stack([
            ## for faces
            torch.cat([torch.tensor([0]), num_faces.cumsum(dim=0)[:-1]]), # start index
            num_faces, # count
            ## for vertices
            torch.cat([torch.tensor([0]), num_verts.cumsum(dim=0)[:-1]]), # start index
            num_verts, # count
        ], dim=1) # (batch_size, 4), on CPU
        return vertices, faces, attrs, ind_group


    @staticmethod
    def save_mesh(path, mesh):
        """ save one mesh
        """
        v, f, a = mesh
        assert a.shape[1] == 3, "attributes should have 3 channels (RGB)"
        if len(v) == 0 or len(f) == 0: # no mesh found
            print(f"mesh is empty, cannot save to: {path}")
            return
        if a.max() < 5: # guess it should be converted into 0~255
            a = (a + 1.0) * 127.5
        mesh = trimesh.Trimesh(
            vertices=v.cpu().numpy(), 
            faces=f.cpu().numpy(), 
            vertex_colors=a.clamp(0, 255).byte().cpu().numpy(),
        )
        mesh.export(path)
        print(f"mesh is saved to: {path}")
    
    
    @staticmethod
    def save_meshes(path, meshes):
        """ save many meshes
        """
        assert "{i" in path # e.g. path == "mesh_{i:05}.ply"
        vv, ff, aa, group = meshes
        for ind_mesh, (f_st, f_cnt, v_st, v_cnt) in enumerate(group):
            v = vv[v_st:(v_st + v_cnt)]
            f = ff[f_st:(f_st + f_cnt)] - v_st
            a = aa[v_st:(v_st + v_cnt)]
            Render.save_mesh(path.format(i=ind_mesh), (v, f, a))


    @staticmethod
    def c2w_to_w2c(c2w_int, c2w_ext):
        """ 
        Args:
            c2w_int.shape == (3, 3) or (..., 3, 3)
            c2w_ext.shape == (3, 4) or (..., 3, 4)
        """
        w2c_int = torch.linalg.inv(c2w_int) # (..., 3, 3)
        rot_inv = rearrange(c2w_ext[..., :3], "... I J -> ... J I")
        w2c_ext = torch.cat([
            rot_inv,
            rot_inv @ c2w_ext[..., [3]].neg(),
        ], dim=-1) # (..., 3, 4)
        return w2c_int, w2c_ext


    def render_many(self, pack_meshes, pack_cameras, res=128):
        """ render many meshes
        Returns:
            img_attr.shape == (B, 3+1, H, W), float32
            img_dep.shape == (B, H, W), float32
            img_vib.shape == (B, H, W), bool
        """
        vertices, faces, attrs, ind_group = pack_meshes
        w2c_int, w2c_ext = self.c2w_to_w2c(*pack_cameras)

        # for each mesh
        f_idx, v_idx = 0, 0
        z_mins, z_maxs = [], []
        f_new, v_new, a_new = [], [], []
        for ind_mesh, (f_st, f_cnt, v_st, v_cnt) in enumerate(ind_group): # TODO it may be slow, but rather straightforward
            f = faces   [f_st:(f_st + f_cnt)] - v_st
            v = vertices[v_st:(v_st + v_cnt)]
            a = attrs   [v_st:(v_st + v_cnt)]
            # project vertices
            v = torch.cat([v, v.new_ones([len(v), 1])], dim=1)
            v = torch.einsum("ik,jk->ij", v, w2c_ext[ind_mesh])
            v = torch.einsum("ik,jk->ij", v, w2c_int[ind_mesh])
            # 
            z  = v[:,  2]
            uv = v[:, :2] / z[:, None]
            uv = (uv - res/2) / (res/2) # to -1 ~ +1
            # 
            msk_v = (z > self.depth_min) & (uv.abs() < 1).all(dim=1) # must be inside the bbox
            msk_f = msk_v[f].all(dim=1) # TODO maybe `any` is better, but compute of `msk_v` is a bit tricky
            # 
            num_v = msk_v.sum().item()
            num_f = msk_f.sum().item()
            # 
            if msk_v.any():
                z_min = z[msk_v].min().item()
                z_max = z[msk_v].max().item()
            else: # when the mesh is empty
                z_min, z_max = 0.0, 1.0
            z_mins.append(z_min)
            z_maxs.append(z_max)
            # 
            w = (z[msk_v] - z_min) / (z_max - z_min) * 2.0 - 1.0 # to -1 ~ +1
            v_new.append(
                torch.cat([uv[msk_v], w[:, None]], dim=1)
            )
            # 
            a_new.append(
                a[msk_v]
            )
            # 
            pi = msk_v.nonzero().squeeze(1)
            p = torch.zeros_like(msk_v, dtype=torch.long)
            p[pi] = torch.arange(len(pi), device=p.device)
            f_new.append(
                p[f[msk_f]] + v_idx
            )
            # 
            ind_group[ind_mesh, 0] = f_idx
            ind_group[ind_mesh, 1] = num_f
            ind_group[ind_mesh, 2] = v_idx
            ind_group[ind_mesh, 3] = num_v
            # 
            f_idx += num_f
            v_idx += num_v
    
        # combine results
        coord_clip = torch.cat(v_new, dim=0)
        coord_clip = torch.cat([coord_clip, coord_clip.new_ones([len(coord_clip), 1])], dim=1)
        faces      = torch.cat(f_new, dim=0)
        attrs      = torch.cat(a_new, dim=0)
        del v_new, f_new, a_new # it may take lots of memory, so delete them

        # depth's range
        z_max = torch.tensor(z_maxs, device=vertices.device)[:, None, None]
        z_min = torch.tensor(z_mins, device=vertices.device)[:, None, None]

        # render
        rast, _ = dr.rasterize(self.ctx_render,
                     coord_clip.float().contiguous(),
                     faces.to(torch.int32).contiguous(),
                     resolution=(res, res),
                     ranges=ind_group[:, :2].to(torch.int32).contiguous(), # has already on CPU
                    )
        # attribute
        img_attr, _ = dr.interpolate(
            attrs.contiguous(),
            rast,
            faces.to(torch.int32).contiguous(),
        )
        img_attr = rearrange(img_attr, "B H W C -> B C H W")
        # depth
        img_dep = (rast[..., 2] + 1.0) / 2.0 # to 0~1
        img_dep = img_dep * (z_max - z_min) + z_min # (B, H, W)
        # visibility
        img_vib = rast.any(dim=3) # bool, (B, H, W)
        return img_attr, img_dep, img_vib


    def unproject_render_many(self, c2w_curr, rgbd_prev, c2w_prev, res=128, trunc=True):
        """ project all the previous views onto the current view
        Args:
            c2w_curr.shape == (B, 3, 3) and (B, 3, 4)
            rgbd_prev.shape == (B, C, H, W)
            c2w_prev.shape == (B, 3, 3) and (B, 3, 4)
        Returns:
            rgbd_out.shape == (B, C, H, W)
            mask_out.shape == (B, H, W)
        """
        meshes_prev = self.meshing_many(rgbd_prev, *c2w_prev, trunc=trunc)
        img_attr, img_dep, mask_out = self.render_many(
            meshes_prev, c2w_curr, res=res,
        )
        # merge
        rgbd_out = torch.cat([img_attr, rearrange(img_dep, "B H W -> B () H W")], dim=1) # (B, C=4, H, W)
        return rgbd_out, mask_out


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from types import SimpleNamespace

import os
os.sys.path = ["./third_party/latent-diffusion"] + os.sys.path
from ldm.modules.diffusionmodules.openaimodel import UNetModel


def get_unet():
    unet = UNetModel(image_size=IMG_SIZE,
                     in_channels=8, out_channels=4,
                     model_channels=128, # the base channel (smallest)
                     channel_mult=[1, 2, 3, 3, 4, 4],
                     num_res_blocks=2, 
                     num_head_channels=32,
                     # down 1     2      4          8          16         32
                     # res  128   64     32         16         8          4
                     # chan 128   256    384        384        512        512
                     # type conv  conv   conv+attn  conv+attn  conv+attn  conv+attn
                     attention_resolutions=[4, 8, 16, 32],
                     use_checkpoint=True,
                     use_fp16=FP16_MODE,
                    )
    # use num_groups==1, to avoid color shift problem
    for name, module in unet.named_modules():
        if isinstance(module, nn.GroupNorm):
            module.num_groups = 1
            print(f"convert GN to LN for module: {name}")
    return unet



class Model(nn.Module, Render):
    def __init__(self):
        nn.Module.__init__(self)
        Render   .__init__(self)
        self.unet = get_unet()
        # dummy params
        self.no_pixel = nn.Parameter(torch.zeros(4))

    def render_views(self, 
                     rgbd_src, # source views (e.g. previous views), shape == (B, N, C, H, W)
                     c2w_src,  # camera params of source views, shape == (B, N, 3, 3/4)
                     c2w_dst,  # project onto this camera view, shape == (B, 3, 3/4)
                    ):
        B, N = rgbd_src.shape[:2]
        rgbd_src = rgbd_src.flatten(0, 1) # (B*N, C, H, W)
        # undo normalization
        mean, std = mean_std.unbind(dim=1)
        mean = mean[None, :, None, None]
        std  = std [None, :, None, None]
        rgbd_src = rgbd_src * std  + mean
        # 
        c2w_src = tuple(mat.flatten(0, 1) for mat in c2w_src)
        vertices, faces, attrs, ind_group_src = self.meshing_many(rgbd_src, *c2w_src)
        # 
        ind_group_src = ind_group_src.reshape(B, N, 4)
        ind_group_dst = ind_group_src.new_zeros([B, 4])
        ind_group_dst[:, 0] = ind_group_src[:, 0, 0]            # face start
        ind_group_dst[:, 1] = ind_group_src[:, :, 1].sum(dim=1) # face count
        ind_group_dst[:, 2] = ind_group_src[:, 0, 2]            # vert start
        ind_group_dst[:, 3] = ind_group_src[:, :, 3].sum(dim=1) # vert count
        # 
        img_attr, img_dep, mask_out = self.render_many(
            (vertices, faces, attrs, ind_group_dst),
            c2w_dst, res=IMG_SIZE,
        )
        rgbd_out = torch.cat([img_attr, img_dep[:, None, ...]], dim=1) # (B, C, H, W)
        # redo normalization
        rgbd_out = (rgbd_out - mean) / std
        # some pixels are empty because they haven't been projected onto
        rgbd_out = rearrange(rgbd_out, "B C H W -> B H W C")
        rgbd_out[~mask_out] = self.no_pixel
        rgbd_out = rearrange(rgbd_out, "B H W C -> B C H W")
        out = SimpleNamespace(rgbd=rgbd_out, mask=mask_out)
        return out # (B, C, H, W) / (B, H, W)
    
    def forward(self, rgbd, cam, t):
        """ 
        Args:
            rgbd.shape == (B, N, C, H, W)
            cam (tuple): include intrinsic and pose matrices, shape == (B, N, 3, 3) / (B, N, 3, 4)
            
            NOTE 
            only curr view (i.e. rgbd[:, -1]) is noised
            N == num of views (prev + curr)
            invalid depth is zero, then will be trimmed
        """
        B, N = rgbd.shape[:2]
        assert N >= 2, "at least one previous view is provided. if there is no previous views, please provide zero"
        camint, camext = cam
        rgbd_render = self.render_views(
               rgbd[:, :-1],
            (camint[:, :-1], camext[:, :-1]),
            (camint[:,  -1], camext[:,  -1]),
        ).rgbd # (B, C, H, W)
        # 
        unet_in = torch.cat([rgbd_render, rgbd[:, -1]], dim=1) # (B, 4 + 4, H, W)
        pred = self.unet(unet_in, t) # (B, C, H, W)
        return pred



# create model
model = Model()

# convert batchnorm
if num_rank > 1:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

import math
from diffusers import DDIMScheduler

def learning_rate_fn(epoch):
    epoch_max = training["epoch_end"]
    factor_min = LR_FINAL / LR_INIT
    if epoch < epoch_max: # cosine learning rate
        return factor_min + 0.5 * (1.0 - factor_min) * (1.0 + math.cos(epoch / epoch_max * math.pi))
    else:
        return factor_min


optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=LR_INIT)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_fn)
diffusion_scheduler = DDIMScheduler(num_train_timesteps=1000, clip_sample=False, set_alpha_to_one=False)

if FP16_MODE:
    scaler = torch.cuda.amp.GradScaler()



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# do data augmentation
def data_augmentation(batch_data): # TODO maybe RandomHorizontalFlip ???
    return batch_data


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


def eval_loss(batch_data, drop_one=0.1, drop_all=0.1, mode="train"):
    assert mode in ("train", "eval")
    
    # perform augmentation
    if mode == "train":
        batch_data = data_augmentation(batch_data)
    
    # get data
    rgbd = batch_data["rgbd"]
    pose = batch_data["pose"]
    intr = batch_data["intr"]

    # sample time steps
    t = torch.randint(
        0, diffusion_scheduler.config.num_train_timesteps, 
        size=(len(rgbd), ), device=rgbd.device,
    )

    # make image noisy
    noise = torch.randn_like(rgbd[:, -1])
    rgbd[:, -1] = diffusion_scheduler.add_noise(rgbd[:, -1], noise, t) # has been noised

    # make it unconditional
    drop_msk = torch.rand(rgbd.shape[:2]) < drop_one # drop some views
    drop_msk[torch.rand(len(rgbd)) < drop_all] = True # drop all
    drop_msk[:, -1] = False # don't drop current view
    rgbd[drop_msk] = 0.0 # drop by setting depth to zero

    # compute loss
    with torch.cuda.amp.autocast(enabled=FP16_MODE):
        pred_noise = model(rgbd, (intr, pose), t)
        # compute loss
        loss_cor_dict = { "loss_color": F.l1_loss(pred_noise[:, :3 ], noise[:, :3 ]) }
        loss_dep_dict = { "loss_depth": F.l1_loss(pred_noise[:, [3]], noise[:, [3]]) }
    
    return { **loss_cor_dict, **loss_dep_dict }




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



def move_to(d, device):
    d_out = dict()
    for k, v in d.items():
        d_out[k] = v.to(device=device) if isinstance(v, torch.Tensor) else v
    return d_out


def combine_loss(loss_dict):
    loss_dict["loss"] = 0.20 * loss_dict["loss_color"] + \
                        0.80 * loss_dict["loss_depth"] # depth is very important
    return loss_dict




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 



def train_step_fn(batch_data):
    model.train()

    # compute loss
    batch_data = move_to(batch_data, device=device)
    loss_dict = eval_loss(batch_data, mode="train")
    loss_dict = combine_loss(loss_dict)

    # update model
    optimizer.zero_grad()
    if FP16_MODE:
        scaler.scale(loss_dict["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_dict["loss"].backward()
        optimizer.step()

    return dict(**{k:v.item() for k, v in loss_dict.items()},
                lr=optimizer.param_groups[0]["lr"])




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from tqdm import tqdm
from collections import defaultdict


@torch.no_grad()
def evaluate_fn(val_loader): # NOTE evaluating the losses on test set
    model.eval()
    
    count = 0.0
    total_loss_dict = defaultdict(float)
    for batch_data in tqdm(val_loader, desc="evaluating"): # each process will compute
        batch_size = len(batch_data["rgbd"])
        
        # compute loss
        batch_data = move_to(batch_data, device=device)
        loss_dict = eval_loss(batch_data, mode="eval")
        loss_dict = combine_loss(loss_dict)

        # update numbers
        count += batch_size
        for k, v in loss_dict.items():
            total_loss_dict[k] += v.item() * batch_size # sum
    
    # average
    return {k : v/count for k, v in total_loss_dict.items()}


