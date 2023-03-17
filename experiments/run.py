# NOTE 
#   you can launch GUI by giving: "--interactive"
#     generate diverse results by varying camera trajectory/randomness
#     you can control the trajectory by button input

import os
import cv2
import json
import math
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from glob import glob
from tqdm import trange
from collections import namedtuple

from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import torch
import gorilla
from einops import rearrange, repeat

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
parser.add_argument("--ind_scenes",  type=str,   default="0") # you can specify "all" to use all scenes
# e.g. 
# "10%"  : only use 10% cameras
parser.add_argument("--task",        type=str,   default="10%")
# seed
parser.add_argument("--seed",        type=int,   default=0)
# trajectory transform
parser.add_argument("--traj",        type=str,   default="")
parser.add_argument("--interactive", action="store_true"   )
# save folder suffix
parser.add_argument("--suffix",      type=str,   default="")
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
seed       = args.seed
traj       = args.traj
suffix     = args.suffix

# 
interactive = args.interactive # interactive terminal control

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


class CameraTrajectoryTransform:
    def __init__(self, camext_lst, ind_dont_change): # NOTE must be adjacent in time sequence
        assert isinstance(camext_lst, torch.Tensor) # (B, 3, 4)
        assert isinstance(ind_dont_change, torch.Tensor)
        self.kwargs = dict(dtype=camext_lst.dtype, device=camext_lst.device)
        self.camext_lst = camext_lst.detach().cpu().numpy() # to numpy
        self.ind_dont_change = ind_dont_change.detach().cpu().numpy() # to numpy

    def interpolate(self, between=5):
        assert between >= 0
        num = len(self.camext_lst)
        # 
        t_in  = np.arange(num)
        t_out = np.linspace(0, num - 1, num + (num - 1) * between)
        # interpolate rotations
        new_R = Slerp(t_in, Rotation.from_matrix(self.camext_lst[:, :, :3]))(t_out).as_matrix() # (num, 3, 3)
        # interpolate translations
        new_T = interp1d(t_in, self.camext_lst[:, :, 3].T)(t_out).T # (num, 3)
        # 
        new_RT = np.concatenate([new_R, new_T[:, :, None]], axis=2)
        ind_insert = torch.from_numpy(t_in * (between + 1)).long() # CPU
        return torch.from_numpy(new_RT).to(**self.kwargs), ind_insert

    def add_noise(self, std_loc=0.2, std_rot=0.1):
        # add noise to rotations
        quat = Rotation.from_matrix(self.camext_lst[:, :, :3]).as_quat()
        quat += np.random.randn(*quat.shape) * std_rot
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        new_R = Rotation.from_quat(quat).as_matrix()
        # add translations
        new_T = self.camext_lst[:, :, 3].copy()
        new_T += np.random.randn(*new_T.shape) * std_loc
        # 
        new_RT = np.concatenate([new_R, new_T[:, :, None]], axis=2)
        new_RT[self.ind_dont_change] = self.camext_lst[self.ind_dont_change] # don't change them
        ind_insert = torch.arange(len(self.camext_lst)).long() # CPU
        return torch.from_numpy(new_RT).to(**self.kwargs), ind_insert




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


class Sampler:
    @torch.no_grad()
    def __call__(self, scene_mesh, cam_curr, seed=seed):
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

PIC_SIZE = 0.25
CTRL_WID = 0.25
    
BTN_HEI = 0.09
BTN_WID = 0.07
BTN_PAD_X = 0.01
BTN_PAD_Y = 0.04

DPI = 100
HEIGHT = 300
WIDTH = int(HEIGHT / PIC_SIZE)

INC_LOC = 0.15
INC_DEG = 10


class Interface:
    def __init__(self, rgbdm, cam, cam_to_rgbdm_fn, cache_folder=None):
        # save
        self.rgbdm = rgbdm # NOTE (H, W, 5)   rgb:0~1   d:meters   m:0/1
        self.cam = cam
        self.cam_to_rgbdm_fn = cam_to_rgbdm_fn
        self.cache_folder = cache_folder # output folder to save temporary results
        # how many times you press the inpaint button, it will change the randomness seed
        self.times_press_inpaint_btn = 0
        # index of fused view
        self.index_view = 0
        # backup, so that we can possibly recover the previous step
        self.back_up = None # NOTE we only store one previous step
        # build GUI
        self.initialize_interface()

    def feed_display(self, what): # get each image prepared for display, output is RGB image in range 0~1
        assert what in ("rgb", "d", "m")
        rgbdm = self.rgbdm.clone() # make sure you don't change it in place
        msk = rgbdm[..., 4] > 0.5
        if what == "rgb":
            mat = rgbdm[..., :3]
            mat[~msk] = 0
            mat = mat.detach().cpu().numpy()
        elif what == "d":
            mat = rgbdm[..., 3].clamp(min=0.1, max=10.0)
            mat[~msk] = 10.0
            mat = 1 / mat # 0.1~10
            perc_3 = torch.quantile(mat.flatten(), 0.03)
            perc_97 = torch.quantile(mat.flatten(), 0.97)
            mat = mat.clamp(min=perc_3, max=perc_97) # remove noise
            mat = (mat - mat.min()) / (mat.max() - mat.min()) # 0~1
            mapper = cm.ScalarMappable(cmap="magma")
            mat = mapper.to_rgba(mat.detach().cpu().numpy())[..., :3]
            mat[~msk.detach().cpu().numpy()] = 0
        else:
            mat = repeat(msk, "H W -> H W C", C=3).float()
            mat = mat.detach().cpu().numpy()
        return mat
    
    def initialize_interface(self): # build the GUI interface
        # 
        self.fig = plt.figure(num='RGBD Diffusion', figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI)
        self.fig.canvas.toolbar.pack_forget() # remove status bar
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # color
        self.ax_color = self.fig.add_axes([0, 0, PIC_SIZE, 1])
        self.ax_color.set_axis_off()
        self.pic_color = self.ax_color.imshow(self.feed_display("rgb"))
        # depth
        self.ax_depth = self.fig.add_axes([PIC_SIZE, 0, PIC_SIZE, 1])
        self.ax_depth.set_axis_off()
        self.pic_depth = self.ax_depth.imshow(self.feed_display("d"))
        # binary mask
        self.ax_mask = self.fig.add_axes([PIC_SIZE * 2, 0, PIC_SIZE, 1])
        self.ax_mask.set_axis_off()
        self.pic_mask = self.ax_mask.imshow(self.feed_display("m"))
        # 
        self.update_pose_and_render(self.cam[1]) # NOTE immediately launch rendering
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # translation control
        self.fig.text(1 - (BTN_WID + BTN_PAD_X) * 3, 0.94, "Translation Control:", fontsize=10)
        # 
        self.ax_move_left = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 3, 1 - (BTN_HEI + BTN_PAD_Y) * 2 - 0.05, BTN_WID, BTN_HEI])
        self.btn_move_left = Button(self.ax_move_left, 'Left')
        self.btn_move_left.on_clicked(self.move_left)
        # # 
        self.ax_move_back = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 3, 1 - (BTN_HEI + BTN_PAD_Y) - 0.05, BTN_WID, BTN_HEI])
        self.btn_move_back = Button(self.ax_move_back, 'Back')
        self.btn_move_back.on_clicked(self.move_back)
        # # 
        self.ax_move_right = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 1, 1 - (BTN_HEI + BTN_PAD_Y) * 2 - 0.05, BTN_WID, BTN_HEI])
        self.btn_move_right = Button(self.ax_move_right, 'Right')
        self.btn_move_right.on_clicked(self.move_right)
        # # 
        self.ax_move_forw = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 1, 1 - (BTN_HEI + BTN_PAD_Y) - 0.05, BTN_WID, BTN_HEI])
        self.btn_move_forw = Button(self.ax_move_forw, 'Forward')
        self.btn_move_forw.on_clicked(self.move_forw)
        # # 
        self.ax_move_down = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 2, 1 - (BTN_HEI + BTN_PAD_Y) * 2 - 0.05, BTN_WID, BTN_HEI])
        self.btn_move_down = Button(self.ax_move_down, 'Down')
        self.btn_move_down.on_clicked(self.move_down)
        # # 
        self.ax_move_up = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 2, 1 - (BTN_HEI + BTN_PAD_Y) - 0.05, BTN_WID, BTN_HEI])
        self.btn_move_up = Button(self.ax_move_up, 'Up')
        self.btn_move_up.on_clicked(self.move_up)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # rotation control
        self.fig.text(1 - (BTN_WID + BTN_PAD_X) * 3, 1.0 - (BTN_HEI + BTN_PAD_Y) * 3, "Rotation Control:", fontsize=10)
        # 
        self.ax_rota_left = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 3, 1 - (BTN_HEI + BTN_PAD_Y) * 5 + 0.01, BTN_WID, BTN_HEI])
        self.btn_rota_left = Button(self.ax_rota_left, 'Left')
        self.btn_rota_left.on_clicked(self.rota_left)
        # # 
        self.ax_rota_down = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 2, 1 - (BTN_HEI + BTN_PAD_Y) * 5 + 0.01, BTN_WID, BTN_HEI])
        self.btn_rota_down = Button(self.ax_rota_down, 'Down')
        self.btn_rota_down.on_clicked(self.rota_down)
        # # 
        self.ax_rota_right = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 1, 1 - (BTN_HEI + BTN_PAD_Y) * 5 + 0.01, BTN_WID, BTN_HEI])
        self.btn_rota_right = Button(self.ax_rota_right, 'Right')
        self.btn_rota_right.on_clicked(self.rota_right)
        # # 
        self.ax_rota_up = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 2, 1 - (BTN_HEI + BTN_PAD_Y) * 4 + 0.01, BTN_WID, BTN_HEI])
        self.btn_rota_up = Button(self.ax_rota_up, 'Up')
        self.btn_rota_up.on_clicked(self.rota_up)
        # 
        self.ax_rota_ccw = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 3, 1 - (BTN_HEI + BTN_PAD_Y) * 4 + 0.01, BTN_WID, BTN_HEI])
        self.btn_rota_ccw = Button(self.ax_rota_ccw, 'CCW')
        self.btn_rota_ccw.on_clicked(self.rota_ccw)
        # # 
        self.ax_rota_cw = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 1, 1 - (BTN_HEI + BTN_PAD_Y) * 4 + 0.01, BTN_WID, BTN_HEI])
        self.btn_rota_cw = Button(self.ax_rota_cw, 'CW')
        self.btn_rota_cw.on_clicked(self.rota_cw)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        # rotation control
        self.fig.text(1 - (BTN_WID + BTN_PAD_X) * 3, 1.05 - (BTN_HEI + BTN_PAD_Y) * 6, "Operation:", fontsize=10)
        # 
        self.ax_inpaint = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 3, 1 - (BTN_HEI + BTN_PAD_Y) * 7 + 0.07, 1.5 * BTN_WID + 0.5 * BTN_PAD_X, BTN_HEI])
        self.btn_inpaint = Button(self.ax_inpaint, 'Inpaint')
        self.btn_inpaint.on_clicked(self.op_inpaint)
        # 
        self.ax_done = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 1.5, 1 - (BTN_HEI + BTN_PAD_Y) * 7 + 0.07, 1.5 * BTN_WID + 0.5 * BTN_PAD_X, BTN_HEI])
        self.btn_done = Button(self.ax_done, 'All Done')
        self.btn_done.on_clicked(self.op_done)
        # 
        self.ax_prev = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 3, 1 - (BTN_HEI + BTN_PAD_Y) * 8 + 0.07, 1.5 * BTN_WID + 0.5 * BTN_PAD_X, BTN_HEI])
        self.btn_prev = Button(self.ax_prev, 'Prev')
        self.btn_prev.on_clicked(self.op_prev)
        # 
        self.ax_next = self.fig.add_axes([1 - (BTN_WID + BTN_PAD_X) * 1.5, 1 - (BTN_HEI + BTN_PAD_Y) * 8 + 0.07, 1.5 * BTN_WID + 0.5 * BTN_PAD_X, BTN_HEI])
        self.btn_next = Button(self.ax_next, 'Next')
        self.btn_next.on_clicked(self.op_next)
        # 

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def decompose_pose(self, camext):
        right  = camext[:, 0]
        up     = camext[:, 1].neg()
        lookat = camext[:, 2]
        loc    = camext[:, 3]
        return namedtuple("CameraCoordSystem", ["right", "up", "lookat", "loc"])(right, up, lookat, loc)

    def display_rgbdm(self): # display to window        
        self.pic_color.set_data(self.feed_display("rgb")) # display color image to GUI
        self.pic_depth.set_data(self.feed_display("d")) # display depth image to GUI
        self.pic_mask.set_data(self.feed_display("m")) # display mask image to GUI

    def update_pose_and_render(self, pose_new):
        self.cam = (self.cam[0], pose_new) # update pose
        self.rgbdm = self.cam_to_rgbdm_fn(self.cam) # render image
        self.display_rgbdm()
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def move_left(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] -= INC_LOC * self.decompose_pose(camext).right
        self.update_pose_and_render(camext_new)
    
    def move_right(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] += INC_LOC * self.decompose_pose(camext).right
        self.update_pose_and_render(camext_new)

    def move_back(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] -= INC_LOC * self.decompose_pose(camext).lookat
        self.update_pose_and_render(camext_new)
    
    def move_forw(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] += INC_LOC * self.decompose_pose(camext).lookat
        self.update_pose_and_render(camext_new)

    def move_down(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] -= INC_LOC * self.decompose_pose(camext).up
        self.update_pose_and_render(camext_new)
    
    def move_up(self, _):
        camext = self.cam[1]
        camext_new = camext.clone()
        camext_new[:, 3] += INC_LOC * self.decompose_pose(camext).up
        self.update_pose_and_render(camext_new)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def rota_left(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(camext[:, :3].detach().cpu().numpy()) * 
                        Rotation.from_euler("y", -INC_DEG, degrees=True)
                    ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    def rota_right(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(camext[:, :3].detach().cpu().numpy()) * 
                        Rotation.from_euler("y", INC_DEG, degrees=True)
                    ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    def rota_down(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(camext[:, :3].detach().cpu().numpy()) * 
                        Rotation.from_euler("x", -INC_DEG, degrees=True)
                    ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)
    
    def rota_up(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(camext[:, :3].detach().cpu().numpy()) * 
                        Rotation.from_euler("x", INC_DEG, degrees=True)
                    ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    def rota_ccw(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(camext[:, :3].detach().cpu().numpy()) * 
                        Rotation.from_euler("z", INC_DEG, degrees=True)
                    ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)
    
    def rota_cw(self, _):
        camext = self.cam[1]
        rotation_new = (Rotation.from_matrix(camext[:, :3].detach().cpu().numpy()) * 
                        Rotation.from_euler("z", -INC_DEG, degrees=True)
                    ).as_matrix().astype("float32")
        camext[:, :3] = torch.from_numpy(rotation_new).to(camext.device)
        self.update_pose_and_render(camext)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

    def op_inpaint(self, _): 
        # NOTE only inpaint and visualize, 
        # but do not supplement to the mesh, 
        # and do not save to file;
        # you can press the button for many times to get desirable results
        # 
        rgbd_curr, mask_known = sample_one_view(
            scene_mesh, self.cam,
            seed=self.times_press_inpaint_btn,
        ) # (H, W, 4)
        # NOTE we inpaint all the pixels, so the mask should be ideally all ones
        mask_known.fill_(1.0)
        # 
        rgbd_curr = rgbd_curr * std + mean
        rgbd_curr[..., :3] = (rgbd_curr[..., :3] + 1) / 2 # 0~1
        rgbd_curr[..., :3] = rgbd_curr[..., :3].clamp(min=0, max=1)
        rgbd_curr[...,  3] = rgbd_curr[...,  3].clamp(min=0)
        # 
        self.rgbdm = torch.cat([
            rgbd_curr, mask_known[..., None].float()
        ], dim=2) # (H, W, 5)
        # 
        self.times_press_inpaint_btn += 1
        self.display_rgbdm() # show
    
    def op_next(self, _):
        # NOTE supplement to the mesh,
        # also save to file
        # 
        global scene_mesh
        self.back_up = dict( # all are in CPU to save memory
            index_view = self.index_view,
            times_press_inpaint_btn = self.times_press_inpaint_btn,
            scene_mesh = [t.cpu().numpy() for t in scene_mesh],
            cam = [t.cpu().numpy() for t in self.cam],
            rgbdm = self.rgbdm.cpu().numpy(),
        )
        # 
        rgbdm = self.rgbdm.clone() # retrieve the just rendered image
        rgbdm[..., :3] = 2 * rgbdm[..., :3] - 1 # -1 ~ 1
        rgbd = rgbdm[..., :4]
        # merge current mesh into `scene_mesh`
        scene_mesh = merge_mesh(scene_mesh,
            model.meshing(rgbd, *self.cam)
        ); scene_mesh = simplify_mesh(scene_mesh)
        # refresh the image immediately
        self.update_pose_and_render(self.cam[1])
        # save to cache folder
        if self.cache_folder is not None:
            # save mesh
            save_mesh(osp.join(self.cache_folder, f"mesh_{self.index_view}.ply"), scene_mesh)
            # save color
            img_color_path = osp.join(self.cache_folder, f"color_{self.index_view}.png")
            cv2.imwrite(img_color_path, 
                    rgbd[..., [2, 1, 0]].add(1.0).mul(127.5).round() \
                                        .clamp(0, 255).cpu().numpy().astype(np.uint8),
            ); print(f"color image is saved to: {img_color_path}")
            # save depth
            img_depth_path = osp.join(self.cache_folder, f"depth_{self.index_view}.png")
            cv2.imwrite(img_depth_path, 
                    rgbd[..., 3].mul(1000.0).round() \
                                .clamp(0, 65535).cpu().numpy().astype(np.uint16),
            ); print(f"depth image is saved to: {img_depth_path}")
            # save camera pose as a json file
            cam_path = osp.join(self.cache_folder, f"camera_{self.index_view}.json")
            with open(cam_path, "w") as f:
                json.dump(dict(
                    camint=self.cam[0].cpu().numpy().tolist(),
                    camext=self.cam[1].cpu().numpy().tolist(),
                ), f, indent=4, sort_keys=True)
            print(f"camera parameter is saved to: {cam_path}")
        # next view
        self.index_view += 1

    def op_prev(self, _): # equivalent to "ctrl+z", go back to the previous result
        if self.back_up is not None:
            global scene_mesh
            # recover others
            self.index_view = self.back_up["index_view"]
            self.times_press_inpaint_btn = self.back_up["times_press_inpaint_btn"]
            scene_mesh = [torch.from_numpy(t).to(device=device) for t in self.back_up["scene_mesh"]]
            self.cam = [torch.from_numpy(t).to(device=device) for t in self.back_up["cam"]]
            self.rgbdm = torch.from_numpy(self.back_up["rgbdm"]).to(device=device)
            # we also remove those saved files
            for p in [osp.join(self.cache_folder, f"mesh_{self.index_view}.ply"),
                      osp.join(self.cache_folder, f"color_{self.index_view}.png"),
                      osp.join(self.cache_folder, f"depth_{self.index_view}.png"),
                      osp.join(self.cache_folder, f"camera_{self.index_view}.json"),
                    ]:
                if osp.exists(p): os.remove(p)
            # clear backup
            self.back_up = None
            # render immediately
            self.update_pose_and_render(self.cam[1])
            print(f"successfully restore a previous step")
        else:
            WARN = "\033[91m[WARNING]\033[0m" # the warning word
            print(f"{WARN} cannot restore the previous step, because there is no backup.")

    def op_done(self, _):
        plt.close() # close the window

    def launch(self):
        plt.show() # pop out the window



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

            # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # NOTE non-interactive
            if not interactive:
                # modify the camera trajectory (pose) here
                if traj != "": # example: "add_noise(std_loc=0.2,std_rot=0.1)|interpolate(between=2)"
                    commands = traj.split("|")
                    new_pose = data["pose"].clone()
                    ind_views_down = torch.tensor(ind_views_down)
                    ind_views_down_before = ind_views_down.clone()
                    for cmd in commands:
                        try:
                            new_pose, ind_insert = eval(f"CameraTrajectoryTransform(new_pose, ind_views_down).{cmd}")
                            ind_views_down = ind_insert[ind_views_down] # get new index
                        except Exception as e:
                            print(f"cannot parse command: {cmd}")
                            print(f"error message: {e}")
                            exit(0)
                    # 
                    num_views = len(new_pose)
                    data["pose"] = new_pose
                    # 
                    intr_original = data["intr"].clone()
                    data["intr"] = intr_original.mean(dim=0, keepdim=True).repeat([num_views, 1, 1]) # NOTE use the average intrinsic matrix
                    data["intr"][ind_views_down] = intr_original[ind_views_down_before] # visible views
                    # 
                    rgbd_original = data["rgbd"].clone()
                    data["rgbd"] = torch.zeros([num_views] + list(rgbd_original.shape[1:]), device=rgbd_original.device)
                    data["rgbd"][ind_views_down] = rgbd_original[ind_views_down_before] # visible views
                    # 
                    ind_views_down = ind_views_down.tolist()
                    num_views_down = len(ind_views_down)
                    # 
                    print(f"UPDATED TRAJECTORY: found {num_views} views in total, but we only use {num_views_down} views actually, due to percentage down-sampling.")
                
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
                            scene_mesh, cam_curr, seed=seed + ind_view,
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
                save_files(osp.join(out_path, scene_name + suffix, "pred"), scene_name, 
                            rgbd_result, mask_result, cam_result, 
                            [(i in ind_views_down) for i in range(num_views)],
                        )

            # # # # # # # # # # # # # # # # # # # # # # # # # # 
            else: # NOTE interactive, you can control the trajectory
                scene_name = f"scene{ind_scenes_process[ind_lst]:04}"
                # prepare input views, and save as files
                rgbd_input_lst, mask_input_lst, cam_input_lst = [], [], []
                for ind_view in ind_views_down:
                    rgbd_input_lst += [rearrange(data["rgbd"][ind_view], "C H W -> H W C")]
                    mask_input_lst += [torch.ones([IMG_SIZE, IMG_SIZE], dtype=torch.bool, device=device)]
                    cam_input_lst  += [(data["intr"][ind_view], data["pose"][ind_view])]
                save_files(osp.join(out_path, scene_name + suffix, "pred", "view_input"), scene_name, 
                            rgbd_input_lst, mask_input_lst, cam_input_lst, 
                            [True for _ in ind_views_down],
                        )
                # 
                cache_folder = osp.join(out_path, scene_name + suffix, "pred", "interactive_output")
                os.makedirs(cache_folder, exist_ok=True)
                # 
                def cam_to_rgbdm_fn(cam_curr):
                    global scene_mesh
                    rgb_out, d_out, vis_out = model.render_many(
                        (*scene_mesh, torch.tensor([[0, len(scene_mesh[1]), 0, len(scene_mesh[0])]])), 
                        (cam_curr[0][None, ...], cam_curr[1][None, ...]),
                        res=IMG_SIZE,
                    )
                    rgb_out = rearrange(rgb_out, "() C H W -> H W C")
                    rgb_out = (rgb_out + 1) / 2 # to 0~1
                    rgbdm = torch.cat([
                        rgb_out, d_out[0, ..., None], vis_out[0, ..., None],
                    ], dim=2) # (H, W, 5)
                    return rgbdm
                # prepare initial image and camera
                init_rgbd = rearrange(data["rgbd"][0], "C H W -> H W C") * std + mean
                init_rgbd[..., :3] = (init_rgbd[..., :3] + 1) / 2 # to 0~1
                init_rgbdm = torch.cat([
                    init_rgbd, torch.ones([IMG_SIZE, IMG_SIZE, 1], dtype=torch.bool, device=device)
                ], dim=2)
                init_cam = [data["intr"][0], data["pose"][0]]
                # buil GUI
                gui = Interface(init_rgbdm, init_cam, cam_to_rgbdm_fn=cam_to_rgbdm_fn, cache_folder=cache_folder)
                gui.launch() # pop out the window

            # save GT
            if save_gt:
                save_files(osp.join(out_path, scene_name + suffix, "gt"), scene_name,
                    [rearrange(rgbd, "C H W -> H W C") for rgbd in data["rgbd"]], 
                    [torch.ones([IMG_SIZE, IMG_SIZE], dtype=torch.bool, device=device)] * num_views,
                    cam_result, [True] * num_views,
                )

            print("= = = = = = = = = = = = = = = = = = = = = = = =")




    else:
        raise







