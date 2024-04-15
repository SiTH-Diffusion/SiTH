# SiTH: Single-view Textured Human Reconstruction with Image-Conditioned Diffusion

## [Project Page](https://ait.ethz.ch/sith) | [Paper](https://arxiv.org/abs/2311.15855) | [Youtube(6min)](https://www.youtube.com/watch?v=gixakzI9UcM)

<img src="assets/teaser.gif" width="800"/> 

Official code release for CVPR 2024 paper [**SiTH**](https://ait.ethz.ch/sith).


What you can find in this repo:
* Demo for reconstructing a fully textured 3D human from a single image in 2 minutes (tested on an RTX 3090 GPU)
* A minimal script for fitting the SMPL-X model to an image.
* A new evaluation benchmark for single-view 3D human reconstruction.
- [ ] [TODO] Training scripts for the diffusion model and the mesh reconstruction model.
- [ ] [TODO] A Gradio demo for creating 3D humans with text prompts.

If you find our code and paper useful, please cite it as
```
@inproceedings{ho2024sith,
    title={SiTH: Single-view Textured Human Reconstruction with Image-Conditioned Diffusion},
    author={Hsuan-I Ho and Jie Song and Otmar Hilliges},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2024}
  }
```

## News
* [April 15, 2024] Release demo code, models, and the evaluation benchmark.



## Installation
Our code has been tested with PyTorch 2.1.0, CUDA 12.1, and an RTX 3090 GPU.

Simply run the following command to install relevant packages:

```bash
pip install -r requirements.txt
```


## Quick Start

1. Download the checkpoint files into the `checkpoints` folder.

```bash
bash tools/download.sh
```

2. Download  [SMPL-X](https://smpl-x.is.tue.mpg.de/) models and move them to the `data/body_models` folder. You should have the following data structure:
```
body_models
    └──smplx
        ├── SMPLX_NEUTRAL.pkl
        ├── SMPLX_NEUTRAL.npz
        ├── SMPLX_MALE.pkl
        ├── SMPLX_MALE.npz
        ├── SMPLX_FEMALE.pkl
        └── SMPLX_FEMALE.npz
```

3. Run the script for body fitting, back hallucination, and mesh reconstruction.
```bash
bash run.sh
```


## SiTH Pipeline

### Data Preparation
You can prepare your own **RGBA** images and put them into the `data/examples/rgba` folder. For example, you can create photos from [OutfitAnyone](https://huggingface.co/spaces/HumanAIGC/OutfitAnyone), and remove the background with [Segment Anything](https://segment-anything.com/demo) or [Clipdrop](https://clipdrop.co/remove-background).

1. Run the script to generate square and centralized input images into the `data/examples/images` folder. The default size is 1024x1024. You can also adjust the size by adjusting the `--size` and `--ratio` arguments. 

```bas!
python tools/centralize_rgba.py
```

2. Install and run [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to get `.json` files of COCO-25 body, hand, and face keypoints. For example, we used the following command, and your image folder should contain files as in [`data/examples/images`](https://github.com/SiTH-Diffusion/SiTH/tree/main/data/examples/images).
```bash!
cd /path/to/openpose_dir

./build/examples/openpose/openpose.bin --image_dir /path/to/images_dir --write_json /path/to/images_dir --display 0 --net_resolution -1x544 --scale_number 3 --scale_gap 0.25 --hand --face --render_pose 0
```

### SMPL-X Fitting

Next, we fit the SMPL-X body model to each input image and align them within a cube of [-1, 1]. By default, we use the following command that optimizes the global orientation, body shape, scale, and X,Y offset parameters. 
```
python fit.py --opt_orient --opt_betas
```
There are also additional arguments and hyperparameters for customized fitting. For example, if you find the initial body pose not perfectly aligned, you can use the `--pot_pose` flag to optimize specific [body joints](https://github.com/SiTH-Diffusion/SiTH/blob/main/fit.py#L55). You can visualize the fitting results by activating the `--debug` flag.


### Back-view Hallucination
Given the front-view images and SMPL-X parameters, we generate back-view images with our image-conditioned diffusion model. The following command generates images in the `data/examples/back_images` folder. 
```
python hallucinate.py --num_validation_image 8
```
Note that generative models do have randomness. Therefore multiple images are generated and you can choose the best one to replace it in `data/examples/back_images`. There are several parameters you can play with:
* `--guidance_scale`: Classifier-free guidance (CFG) scale.
* `--conditioning_scale`: ControlNet conditioning scale.
* `--num_inference_steps`: Denoising steps.
* `--pretrained_model_name_or_path`: The default model is trained on 500 human scans. We offer a new model trained with 2000+ scans and more view angles. To use the model, please adjust to `hohs/SiTH-diffusion-2000`.

### Textured Human Reconstruction
Before reconstructing the 3D meshes, make sure the following folders and images are ready.
```
data/examples
    ├──images
    |   ├── 000.png
    |   ├── 000_keypoints.json
    |   ...
    |
    ├──smplx
    |   ├── 000_smplx.obj
    |   ...
    |
    └──back_images
        ├── 000_00X.png
        ...
```

The following command will reconstruct textured meshes under `data/example/meshes`:

```
python reconstruct.py --test-folder data/example --config recon/config.yaml --resume checkpoints/recon_model.pth
```
The default `--grid_size` for marching cube is set to 512. If your images contain noisy segmentation borders, you can increase `--erode_iter` to shrink your segmentation mask. 


## Evaluation Benchmark

We created an evaluation benchmark using the [CustomHumans](https://custom-humans.github.io/#download) dataset. Please apply the dataset directly and you will find the necessary files in the download link. 

Note that we trained our models with 526 human scans provided in the [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset) dataset and tested on 60 scans in the [CustomHumans](https://custom-humans.github.io/#download) dataset. We used the default hyperparameters and commands suggested in `run.sh`.

## Acknowledgement
We used code from other great research work, including [occupancy_networks](https://github.com/autonomousvision/occupancy_networks),
[pifuhd](https://github.com/facebookresearch/pifuhd), [kaolin-wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp), [mmpose](https://github.com/open-mmlab/mmpose), [smplx](https://github.com/vchoutas/smplx), [SMPLer-X](https://github.com/caizhongang/SMPLer-X), [editable-humans](https://github.com/custom-humans/editable-humans).

We created all the videos using powerful [aitviewer](https://eth-ait.github.io/aitviewer/).

We sincerely thank the authors for their awesome work!
