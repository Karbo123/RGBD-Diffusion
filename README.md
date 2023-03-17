<div align="center">

  <h1 align="center">RGBD<sup>2</sup>: Generative Scene Synthesis via Incremental <br> View Inpainting using RGBD Diffusion Models</h1>
  
  <div align="center" style="text-decoration: none;">
    <a href="https://jblei.site/"><b>Jiabao Lei</b></a>
    ·
    <a href="https://tangjiapeng.github.io/"><b>Jiapeng Tang</b></a>
    ·
    <a href="http://kuijia.site/"><b>Kui Jia</b></a>
  </div>
  
  <h2 align="center">CVPR 2023</h2>
  <div align="center">
    <br>
    <img src="https://i.328888.xyz/2023/03/17/K9yTA.jpeg">
  </div>
  
  <h3 align="center">
    In this work, we present a new solution termed RGBD<sup>2</sup> that sequentially <br> 
    generates novel RGBD views along a camera trajectory, and the scene geometry <br>
    is simply the fusion result of these views.
  </h3>
  
  <p align="center"><br>
    <a href="https://arxiv.org/abs/2212.05993"> <img src='https://img.shields.io/badge/Paper-PDF-green?style=for-the-badge&logo=adobeacrobatreader&logoWidth=20&logoColor=white&labelColor=66cc00&color=94DD15' alt='Paper PDF'>
    </a>
    <a href='https://jblei.site/proj/rgbd-diffusion'> <img src='https://img.shields.io/badge/Project-Page-orange?style=for-the-badge&logo=Google%20chrome&logoColor=white&labelColor=D35400' alt='Project Page'>
    </a>
  </p>

</div>



# Preparation

```bash
# download this repo
git clone git@github.com:Karbo123/RGBD-Diffusion.git --depth=1
cd RGBD-Diffusion
git submodule update --init --recursive

# set up environment
conda create -n RGBD2 python=3.8
conda activate RGBD2

# install packages
pip install torch # tested on 1.12.1+cu116
pip install torchvision
pip install matplotlib # tested on 3.5.3
pip install opencv-python einops trimesh diffusers ninja open3d

# install dependencies
cd ./third_party/nvdiffrast && pip install . && cd ../..
cd ./third_party/recon      && pip install . && cd ../..

```

Download some files:
1. [the preprocessed ScanNetV2 dataset](https://drive.google.com/file/d/12MUFPsLxJakr5bnLO5XsyGQ4lEN9q2Wb/view?usp=share_link). Extract via `mkdir data_file && unzip scans_keyframe.zip -d data_file && mv data_file/scans_keyframe data_file/ScanNetV2`.
2. [model checkpoint](https://drive.google.com/file/d/1R2fvrnVx4ORh3d9Z5n_NHf97X93S78vo/view?usp=share_link). Extract via `mkdir -p out/RGBD2/checkpoint && unzip model.zip -d out/RGBD2/checkpoint`.

Copy the config file to an output folder:
```
mkdir -p out/RGBD2/backup/config
cp ./config/cfg_RGBD2.py out/RGBD2/backup/config
```

# Training

We provide a checkpoint, so you actually don't need to train a model from scratch.
To launch training, simply run:
```bash
CUDA_VISIBLE_DEVICES=0 python -m recon.runner.train --cfg config/cfg_RGBD2.py
```
If you want to train with multiple GPUs, try setting, e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3`. 
We note that it visualizes the training process by producing some TensorBoard files.


# Inference

To generate a test scene, simply run:
```bash
CUDA_VISIBLE_DEVICES=0 python experiments/run.py
```

By additionally providing `--interactive`, you can control the generation process via manual control using a GUI.
Our GUI code uses Matplotlib, so you can even run the code on a remote server, and use x-server (e.g. MobaXterm) to enable graphic control!

![GUI](https://i.328888.xyz/2023/03/17/LD7h3.png)


# About

If you find our work useful, please consider citing our paper:

```
@InProceedings{Lei_2023_CVPR,
    author = {Lei, Jiabao and Tang, Jiapeng and Jia, Kui},
    title = {RGBD2: Generative Scene Synthesis via Incremental View Inpainting using RGBD Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2023}
}
```


This repo is yet an **early-access** version which is under active update.

If you have any questions or needs, feel free to contact me, or just create a GitHub issue.

