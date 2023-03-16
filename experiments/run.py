
import os
import cv2
import math
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from glob import glob
from tqdm import trange

import torch
import gorilla
from einops import rearrange

import sys; sys.path.append(".")
from dataset import ScanNetScene


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# args
parser = argparse.ArgumentParser()
parser.add_argument("--dir_ckpt",    type=str,   default="./out/RGBD2")
parser.add_argument("--out_path",    type=str,   default="./experiments/out")
parser.add_argument("--model_name",  type=str,   default="model.pt")
parser.add_argument("--guidance",    type=float, default=1.0)
parser.add_argument("--inpainting",  type=str,   default="True")
parser.add_argument("--save_gt",     type=str,   default="False")
parser.add_argument("--strict",      type=str,   default="True") # if strict, don't allow empty rendering
parser.add_argument("--num_steps",   type=int,   default=50)
parser.add_argument("--voxel_size",  type=float, default=0.02) # voxel size of mesh simplification, set to 0.0 to disable
# only those with num_views >= 50 are used
parser.add_argument("--min_views",   type=int,   default=50)
# only perform on subset
parser.add_argument("--ind_scenes",  type=str,   default="all")
# e.g. 
# "10%"  : only use 10% cameras
parser.add_argument("--task",        type=str,   default="10%")
args = parser.parse_args()

# read
dir_ckpt   = osp.abspath(args.dir_ckpt)
out_path   = osp.abspath(args.out_path)
model_name = args.model_name
guidance   = args.guidance
inpainting = bool(eval(args.inpainting))
save_gt    = bool(eval(args.save_gt))
strict     = bool(eval(args.strict))
num_steps  = args.num_steps
voxel_size = args.voxel_size
min_views  = args.min_views
ind_scenes = None if args.ind_scenes.lower() == "all" else \
                    [int(x) for x in args.ind_scenes.split(",")]
task       = args.task

# device
device = torch.device("cuda:0")

# config
cfg_path = glob(osp.join(dir_ckpt, "backup", "config", "cfg_*.py"))
assert len(cfg_path) == 1
cfg_path = cfg_path[0]

# make a dir
os.makedirs(out_path, exist_ok=True)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# load config
cfg = gorilla.Config.fromfile(cfg_path)
diffusion_scheduler = cfg["diffusion_scheduler"]
move_to = cfg["move_to"]
IMG_SIZE = cfg["IMG_SIZE"]
FP16_MODE = cfg["FP16_MODE"]
TRAIN_SPLIT_RATIO = cfg["TRAIN_SPLIT_RATIO"]


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# load dataset
dataset = ScanNetScene(
    path="./data_file/ScanNetV2",
    num_views_sample=None,
    max_len_seq=None,
    img_size=IMG_SIZE,
    normalize=True,
    inter_mode="nearest",
    subset_indices=1.0 - TRAIN_SPLIT_RATIO,
    reverse_frames=False,
)

mean, std = dataset.normalize_mean_std.to(device=device).unbind(dim=1)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# load model
model = cfg["model"]
gorilla.resume(model=model, 
               filename=osp.join(dir_ckpt, "checkpoint", model_name),
               map_location=device,
            )
model.to(device=device)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


class Sampler:
    @torch.no_grad()
    def __call__(self, scene_mesh, cam_curr, seed=0):
        """ Sample A Novel View for One Scene
        Args:
            scene_mesh (tuple): tuple of `vertices`, `faces`, `colors`
            cam_curr (tuple): camera intrinsic and pose matrices, shape==(3, 3/4)
        """
        # sample noise
        kwargs_rand = lambda seed, device=device: dict(generator=torch.Generator(device).manual_seed(seed), device=device)
        z_t = torch.randn([1, 4, IMG_SIZE, IMG_SIZE], **kwargs_rand(seed))
        # compute the known part by rendering the mesh onto the current view
        try:
            rgb_out, d_out, vis_out = model.render_many(
                (*scene_mesh, torch.tensor([[0, len(scene_mesh[1]), 0, len(scene_mesh[0])]])), 
                (cam_curr[0][None, ...], cam_curr[1][None, ...]),
                res=IMG_SIZE,
            )
        except Exception as e:
            if strict: raise e
            print(f"[WARNING] cannot render, because of: {e}")
            rgb_out = torch.zeros([1, 3, IMG_SIZE, IMG_SIZE], device=device)
            d_out   = torch.zeros([1,    IMG_SIZE, IMG_SIZE], device=device)
            vis_out = torch.zeros([1,    IMG_SIZE, IMG_SIZE], device=device)
        rgbd_out = torch.cat([rgb_out, d_out[:, None]], dim=1) # (1, 4, H, W)
        known_part = (rgbd_out - mean[None, :, None, None]) / std[None, :, None, None] # (1, 4, H, W)
        known_part_msk = vis_out[:, None].float() # (1, 1, H, W)
        # 
        known_part = rearrange(known_part, "B C H W -> B H W C")
        known_part[~known_part_msk[0].bool()] = model.no_pixel
        known_part = rearrange(known_part, "B H W C -> B C H W")
        # 
        diffusion_scheduler.set_timesteps(num_steps)
        # 
        time_step_lst = diffusion_scheduler.timesteps
        num_steps_train = diffusion_scheduler.config.num_train_timesteps
        assert num_steps == diffusion_scheduler.num_inference_steps == len(time_step_lst)
        seeds = torch.randint(0, int(1e6), size=(num_steps - 1, ), **kwargs_rand(seed + 1, device="cpu"))
        # 
        model.eval()
        for i, t in enumerate(time_step_lst):
            rgbd_in = torch.cat([known_part, z_t], dim=1) # (1, 8, H, W)
            pred_noise = self.sampling_forward_fn(rgbd_in, t.to(device=device))
            z_t = diffusion_scheduler.step(pred_noise, t.to(device=device), z_t).prev_sample
            # add noise to masking region
            if i < num_steps - 1:
                prev_timestep = t - num_steps_train // num_steps
                z_t_known = diffusion_scheduler.add_noise( # add random noise
                    known_part, 
                    noise=torch.randn(known_part.shape, **kwargs_rand(seeds[i].item())),
                    timesteps=prev_timestep.reshape([1]),
                )
            else: z_t_known = known_part # the last step, use the previous RGBD, don't add noise anymore
            # do masking
            if inpainting:
                z_t = z_t       * (1.0 - known_part_msk) + \
                      z_t_known *        known_part_msk
        # reshape to (H, W, C)
        return rearrange(z_t, "() C H W -> H W C"), known_part_msk[0, 0].bool()

    @torch.no_grad()
    def sampling_forward_fn(self, rgbd, t): # forward func only for classifier-free sampling
        t = t.reshape([1])
        # forward
        with torch.cuda.amp.autocast(enabled=FP16_MODE):
            if guidance == 0.0:
                rgbd[:, :4, ...] = model.no_pixel[None, :, None, None]
                pred = model.unet(rgbd, t)
            elif guidance == 1.0:
                pred = model.unet(rgbd, t)
            else:
                pred_cond   = model.unet(rgbd, t)
                rgbd[:, :4, ...] = model.no_pixel[None, :, None, None]
                pred_uncond = model.unet(rgbd, t)
                pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        return pred.float()


def simplify_mesh(mesh):
    if voxel_size <= 0.0: return mesh
    device = mesh[0].device
    v, f, a = [item.cpu().numpy() for item in mesh]
    a = (a + 1.0) / 2.0 # to 0 ~ 1
    dtype_v = v.dtype
    dtype_f = f.dtype
    dtype_a = a.dtype
    m = o3d.geometry.TriangleMesh()
    m.vertices      = o3d.utility.Vector3dVector(v.astype(np.float64))
    m.triangles     = o3d.utility.Vector3iVector(f.astype(np.int32  ))
    m.vertex_colors = o3d.utility.Vector3dVector(a.astype(np.float64)) 
    m = m.simplify_vertex_clustering(voxel_size=voxel_size)
    v = np.asarray(m.vertices     ).astype(dtype_v)
    f = np.asarray(m.triangles    ).astype(dtype_f)
    a = np.asarray(m.vertex_colors).astype(dtype_a)
    a = a * 2.0 - 1.0 # to -1 ~ +1
    v = torch.from_numpy(v).to(device=device)
    f = torch.from_numpy(f).to(device=device)
    a = torch.from_numpy(a).to(device=device)
    return v, f, a


def merge_mesh(mesh1, mesh2):
    v1, f1, a1 = mesh1
    v2, f2, a2 = mesh2
    v = torch.cat([v1, v2], dim=0)
    f = torch.cat([f1, f2 + len(v1)], dim=0)
    a = torch.cat([a1, a2], dim=0)
    return v, f, a


def empty_mesh():
    v = torch.zeros([0, 3], dtype=torch.float32, device=device)
    f = torch.zeros([0, 3], dtype=torch.long   , device=device)
    a = torch.zeros([0, 3], dtype=torch.float32, device=device)
    return v, f, a


def save_mesh(path, mesh):
    WARN = "\033[91m[WARNING]\033[0m" # the warning word
    if all(len(x) > 0 for x in mesh):
        model.save_mesh(path, mesh)
    else:
        print(f"{WARN} found mesh: {path} was empty, so we didn't save it.")


def save_files(save_folder, scene_name, rgbd_lst, mask_lst, cam_lst, known_lst):
    os.makedirs(save_folder, exist_ok=True)
    all_mesh_lst = list()
    for idx_view, (rgbd_i, mask_i, cam_i, known_i) in enumerate(zip(rgbd_lst, mask_lst, cam_lst, known_lst)):
        rgbd_i = rgbd_i * std + mean
        # 
        mesh_i = model.meshing(rgbd_i, *cam_i)
        all_mesh_lst.append(mesh_i)
        # 
        mark = "known" if known_i else "generation" # if "known", it means that it is a given view (don't need to generate)
        mesh_i_name = f"{scene_name}_view{idx_view:03}_{mark}.ply"
        img_color_i_name = f"{scene_name}_view{idx_view:03}_color_{mark}.png"
        img_depth_i_name = f"{scene_name}_view{idx_view:03}_depth_{mark}.png"
        img_mask_i_name  = f"{scene_name}_view{idx_view:03}_mask_{mark}.png"
        # 
        save_mesh(osp.join(save_folder, mesh_i_name), mesh_i)
        # 
        img_color_i_path = osp.join(save_folder, img_color_i_name)
        cv2.imwrite(img_color_i_path, 
                rgbd_i[..., [2, 1, 0]].add(1.0).mul(127.5).round() \
                                      .clamp(0, 255).cpu().numpy().astype(np.uint8),
        )
        print(f"color image is saved to: {img_color_i_path}")
        # 
        img_depth_i_path = osp.join(save_folder, img_depth_i_name)
        cv2.imwrite(img_depth_i_path, 
                rgbd_i[..., 3].mul(1000.0).round() \
                              .clamp(0, 65535).cpu().numpy().astype(np.uint16),
        )
        print(f"depth image is saved to: {img_depth_i_path}")
        # 
        img_mask_i_path = osp.join(save_folder, img_mask_i_name)
        cv2.imwrite(img_mask_i_path, mask_i.byte().mul(255).cpu().numpy())
        print(f"mask image is saved to: {img_mask_i_path}")

    # save combined mesh
    mesh_combined = all_mesh_lst[0]
    for m in all_mesh_lst[1:]:
        mesh_combined = merge_mesh(mesh_combined, m)
        mesh_combined = simplify_mesh(mesh_combined)
    save_mesh(osp.join(save_folder, f"{scene_name}.ply"), mesh_combined)





# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


if __name__ == "__main__":
    sample_one_view = Sampler()

    if "%" in task: # the percentage task
        perc = float(task.replace("%", "")) / 100.0 # 0~1
        assert 0 < perc <= 1

        # explore dataset
        num_scenes = len(dataset.info_chunk)
        ind_scenes_process = list()
        for ind_sce in range(num_scenes):
            if len(dataset.info_chunk[ind_sce]) >= min_views:
                ind_scenes_process.append(ind_sce)
        num_scenes_process = len(ind_scenes_process)
        print(f"there are {num_scenes} scenes in testset")
        print(f"but only {num_scenes_process} scenes have num_views >= {min_views}")
        print(f"their scene indices are: {ind_scenes_process}")

        # check if indices are out of bound
        if isinstance(ind_scenes, list):
            for ind in ind_scenes:
                assert 0 <= ind < num_scenes_process, \
                    f"only {num_scenes_process} scenes are available, " \
                    f"but you provide index = {ind}"
        
        # for each scene
        for ind_lst in (ind_scenes or range(num_scenes_process)): 
            data = move_to(dataset[ind_scenes_process[ind_lst]], device=device)
            # 
            num_views = len(data["rgbd"])
            num_views_down = math.ceil(num_views * perc)
            print(f"found {num_views} views in total, but we only use {num_views_down} views actually, due to percentage down-sampling.")
            ind_views_down = torch.linspace(0, num_views - 1, num_views_down).round().long().tolist()

            # first build the initial mesh using known views
            scene_mesh = empty_mesh()
            for ind_view in ind_views_down: # for each view
                scene_mesh = merge_mesh(scene_mesh,
                    model.meshing(
                        rearrange(data["rgbd"][ind_view], "C H W -> H W C") * std + mean,
                        data["intr"][ind_view], data["pose"][ind_view],
                    )
                )
                scene_mesh = simplify_mesh(scene_mesh)

            # result storage
            rgbd_result = [None for _ in range(num_views)]
            mask_result = [None for _ in range(num_views)]

            # perform iteration
            for ind_view in trange(num_views, desc="moving camera"):
                if ind_view in ind_views_down: # already exists
                    rgbd_result[ind_view] = rearrange(data["rgbd"][ind_view], "C H W -> H W C")
                    mask_result[ind_view] = torch.ones([IMG_SIZE, IMG_SIZE], dtype=torch.bool, device=device)
                else: # sampling for the novel view
                    cam_curr = (data["intr"][ind_view], data["pose"][ind_view])
                    rgbd_curr, mask_known = sample_one_view(
                        scene_mesh, cam_curr, seed=ind_view,
                    ) # (H, W, 4)
                    
                    # save current view
                    rgbd_result[ind_view] = rgbd_curr
                    mask_result[ind_view] = mask_known

                    # merge current mesh into `scene_mesh`
                    scene_mesh = merge_mesh(scene_mesh,
                        model.meshing(
                            rgbd_curr * std + mean,
                            *cam_curr
                        )
                    )
                    scene_mesh = simplify_mesh(scene_mesh)
            
            # info
            cam_result = list(zip(data["intr"], data["pose"]))
            scene_name = f"scene{ind_scenes_process[ind_lst]:04}"
            
            # save pred
            save_files(osp.join(out_path, scene_name, "pred"), scene_name, 
                        rgbd_result, mask_result, cam_result, 
                        [(i in ind_views_down) for i in range(num_views)],
                    )

            # save GT
            if save_gt:
                save_files(osp.join(out_path, scene_name, "gt"), scene_name,
                    [rearrange(rgbd, "C H W -> H W C") for rgbd in data["rgbd"]], 
                    [torch.ones([IMG_SIZE, IMG_SIZE], dtype=torch.bool, device=device)] * num_views,
                    cam_result, [True] * num_views,
                )

            print("= = = = = = = = = = = = = = = = = = = = = = = =")




    else:
        raise







