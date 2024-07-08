## Model Training

Make sure you run the training script in the `diffusion` directory:

```
cd diffusion
python train.py --config config.yaml [other options]
```

You can expand arguments in the config.py file. The command line arguments will override the parameters in `config.yaml`

## Training options

Here are some important configurations for training:
* `--resolution`: Resolution is set to `512` to save VRAM usage.
* `--sample_random_views`: If `False`, it will only generate training pairs with front and back images. If `True`, two camera viewpoints will be randomly sampled for training. 
* `--white_background`: If `False`, it will randomly add background colors to the image. 
* `--conditioning_channels`: If `4`, will use the UV image and the silhouette mask as ControlNet inputs. If `8`, will use additional viewpoint information as ControlNet inputs.
* `--validation`: Log training images to wandb. 
* `--test`: Test the `data/examples` during training. You need to first fit the SMPL-X parameters for inference. 

## Pretrained Models

We provide some pretrained models:

| Link | Training samples | Resolution | sample_random_views | white_background | conditioning_channels |
| :------: | :----:  | :-----: | :----: | :----: | :----:  |
| [hohs/SiTH_diffusion](https://huggingface.co/hohs/SiTH_diffusion) | 526 | 512 | True | False | 8 |
| [hohs/SiTH-diffusion-2000](https://huggingface.co/hohs/SiTH-diffusion-2000) | 2000 | 512 | True | False | 8 |
| [hohs/SiTH-diffusion-1K](https://huggingface.co/hohs/SiTH-diffusion-1K) | 4000 | 1024 | False | False | 8 |