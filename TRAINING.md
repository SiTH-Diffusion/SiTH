<center> <img src="https://files.ait.ethz.ch/projects/SiTH/sith.png" width=400/> </center>

# <p align="center"> Single-view Textured Human Reconstruction with Image-Conditioned Diffusion </p>

# Model Training Instructions

In the following section, you will find instructions on how to train the models from scratch. In our CVPR submission, we used only 526 3D scans from the [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset) dataset. Recently they have released **THuman2.1** with 2000 more scans. You are welcome to train our method with more data samples and improve its performance. 

## Data Preparation

Please download and unzip the **THuman2.0** dataset (ID 0000~0525 in **THuman2.1**) and its SMPL-X registration to the data folder:

```
data/THuman
    ├──THuman2.0
    |   ├── 0000/
    |   |   ├── 0000.obj
    |   |   ├── material0.jpeg
    |   ...
    |
    └──THuman2.0_smplx
        ├── 0000/smplx_param.pkl
        ...
```

The scans in THuman2.0 have variant scales and orientations. To ensure every training sample is centered within [-1, 1], we need to convert them to the SMPL-X (male) coordinate. We provide the following script to align the training scans:

```
python tools/align_thuman.py
```

Since on-the-fly 3D sampling and rendering is slow, we need to cache data for training. We can generate an h5df dataset file by rasterizing images and sampling 3D points within [-1, 1]. The following command will create a .h5 file around 100GB. 

```
python build_dataset.py
```
We used the following dataset configuration for training:


* `--size`: Image resolution. Default = `1024`.
* `--nsamples`: We will sample `--nsamples` x 4 3D points for training. Default = `1M`.
* `--nviews`: Number of camera views in a uniform circle. Default = `36`.

## Training Hallucination Models

Please see [diffusion/README.md](https://github.com/SiTH-Diffusion/SiTH/blob/main/diffusion/README.md).

## Training Reconstruction Models

Please see [recon/README.md](https://github.com/SiTH-Diffusion/SiTH/blob/main/recon/README.md).