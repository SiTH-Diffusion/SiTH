<p align="center"> <img src="https://files.ait.ethz.ch/projects/SiTH/sith.png" width=400/> </p>

# <p align="center"> Single-view Textured Human Reconstruction with Image-Conditioned Diffusion </p>


## Evaluation Benchmark

We created an evaluation benchmark using the [CustomHumans](https://custom-humans.github.io/#download) dataset. Please apply the dataset directly and you will find the necessary files in the download link. The openpose keypoint files can be downloaded here [60_images](https://files.ait.ethz.ch/projects/SiTH/openpose_60.zip) and [240_images](https://files.ait.ethz.ch/projects/SiTH/openpose_240.zip).

Note that we trained our models with 526 human scans provided in the [THuman2.0](https://github.com/ytrock/THuman2.0-Dataset) dataset and tested on 60 scans in the [CustomHumans](https://custom-humans.github.io/#download) dataset. We used the default hyperparameters and commands suggested in `run.sh`. The evaluation script can be found [here](https://github.com/SiTH-Diffusion/SiTH/blob/main/tools/evaluate.py) and [here](https://github.com/SiTH-Diffusion/SiTH/blob/main/tools/evaluate_image.py). You will need to install two additional packages for evaluation:

```
pip install torchmetrics[image] mediapipe
```

### Single-view human 3D reconstruction benchmark

    
| Methods | P-to-S (cm) ↓ | S-to-P (cm) ↓ | NC ↑ | f-Score ↑ |
| ------  | :----:  | :-----: | :----: | :----: |
| PIFu [[Saito2019]](https://github.com/shunsukesaito/PIFu)   | 2.209 |  2.582  |  0.805  | 34.881 |
| PIFuHD[[Saito2020]](https://github.com/facebookresearch/pifuhd)  | 2.107 |  <ins>2.228</ins>  |  0.804  | **39.076** |
| PaMIR [[Zheng2021]](https://github.com/ZhengZerong/PaMIR)  | 2.181 |  2.507 | <ins>0.813</ins> | 35.847 |
| FOF [[Feng2022]](https://github.com/fengq1a0/FOF)   | <ins>2.079</ins> | 2.644 | 0.808 | 36.013 |
| 2K2K [[Han2023]](https://github.com/SangHunHan92/2K2K) | 2.488 | 3.292 | 0.796 | 30.186 |
| ICON* [[Xiu2022]](https://github.com/YuliangXiu/ICON)  | 2.256 | 2.795 |0.791 | 30.437 |
| ECON* [[Xiu2023]](https://github.com/fengq1a0/FOF)   | 2.483 | 2.680 | 0.797 | 30.894 |
| SiTH* (Ours) | **1.871** | **2.045** | **0.826** | <ins>37.029</ins> | 

* *indicates methods trained on the same THuman2.0 dataset.

### Back-view hallucination benchmark

| Methods | SSIM ↑ | LPIPS↓ | KID(×10^−3^) ↓ | Joints Err. (pixel) ↓ |
| ------  | :----:  | :-----: | :----: | :----: |
| Pix2PixHD [[Wang2018]](https://github.com/NVIDIA/pix2pixHD) |  0.816 | 0.141 | 86.2 | 53.1 |
| DreamPose [[Karras2023]](https://github.com/johannakarras/DreamPose) |  0.844 | 0.132 | 86.7 | 76.7 |
| Zero-1-to-3 [[Liu2023]](https://github.com/cvlab-columbia/zero123)  | <ins>0.862</ins> | <ins>0.119</ins> | <ins>30.0</ins> | 73.4 |
| ControlNet [[Zhang2023]](https://github.com/lllyasviel/ControlNet-v1-1-nightly)   | 0.851 | 0.202 | 39.0 | <ins>35.7</ins> |
| SiTH  (Ours)  | **0.950** | **0.063** | **3.2** | **21.5** |

## How to evaluate your method?

### 1. Single-view human 3D reconstruction benchmark:
The input to your methods is `images_60` and the ground truth is `gt_meshes_60`.
Please note that you **cannot use the ground-truth SMPL-X** meshes as inputs to your method.
Your method should predict the SMPL-X meshes from the input images only.
To avoid scale and depth ambiguity, we use **ICP** to align the predicted meshes to the ground-truth meshes.
You can use the evaluation script by running the following command:

```bash
python tools/evaluate.py -i /path/to/your/results -g /path/to/ground-truth
```

### 2. Back-view hallucination benchmark
The input to your methods is `images_60` and the ground truth is `gt_back_60`.
Note that to consider the stochastic nature of generative models, we generate 16 samples per input image.

```bash
python tools/evaluate_image.py -i /path/to/your/results -g /path/to/ground-truth --mask
```

## How to properly run SiTH with your own data?

1. Please note that SiTH uses an **orthogonal coordinate** system as other methods in the baseline table. If you want to compute 3D metrics, you need to render images with an orthographic camera. 
2. The image resolution for **3D reconstruction and SMPL-X fitting is 1024x1024**. Only the resolution for the diffusion hallucination is 512x512. By default, you should prepare an image with a **1024x1024** resolution, and the config file will handle everything for you. 
3. SiTH is a SMPL-centric pipeline and does not handle special cases like holding an object or children. 
