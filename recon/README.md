## Model Training

Make sure you run the training script in the `recons` directory:

```
cd recon
```

The human reconstruction model is trained in 3 steps:

### Step 1. Training Normal Predictor

We first train a normal predictor using the ground-truth image-normal pairs.

```
python train_normal.py --config config.yaml --wandb --use_mask
```
The `--use_mask` option will compute the L1 loss only in the foreground human area. We train this step for 600 epochs. 

### Step 2. Training Neural Fields w/ GT Normal

In the second step, we use the ground-truth normal images to train the neural fields:

```
python train_full.py --config config.yaml --wandb --valid --valid_folder ../data/examples/ --resume /model/path/to/step_1    
```

Note that you need to load the checkpoint from step 1 with the `--resume` option. We train this step for 400 epochs. 

To run validation during training, you need to run `fit.py` and `hallucinate.py` before inference. 

### Step 3. Finetune both models w/ predicted Normal

Finally, we need to make sure the neural fields can also handle predicted normal images during inference.

```
python train_full.py --config config.yaml --wandb --valid --valid_folder ../data/examples/ --resume /model/path/to/step_1 --use_pred_nrm --lr_decoder 0.0001 --lr_encoder: 0.00001
```

The `--use_pred_nrm` option will finetune both the normal predictor and the neural fields together. We train this step for 400 epochs. 

## Training options

Here are some important configurations for training:
* `--image_size`: The image resolution is set to `1024` to ensure high-quality reconstruction. The hallucinated images will be up-scaled to this resolution. 
* `--use_mask`: If `True`, it will compute the L1 normal loss only in the foreground human area. Using this in the first step can reduce artifacts. 
* `--white_bg`: If `False`, it will randomly add background colors to the image. 
* `--aug_jitter`: Whether to use data augmentation with jittering. This option can be used in step 1.
* `--use_pred_nrm`: If `False`, the neural fields will use GT normal images for 3D reconstruction. The gradients will not back-prop to the normal predictor. This flag is activated in step 3. 
