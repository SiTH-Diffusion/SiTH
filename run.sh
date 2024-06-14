#!/bin/bash

# Prepare your RGBA images in data/examples/rgba
# Step 0: Convert RGBA image to square images and estimate 2D keypoints
# python tools/centralize_rgba.py
# ./build/examples/openpose/openpose.bin ......

# Step 1: Fit SMPL-X to the input image, the result will be saved in data/examples/smplx
python fit.py --opt_orient --opt_betas

# Step 2: Hallucinated back-view images, the result will be saved in data/examples/back_images
# Note that the hallucination process is stochastic, therefore you may choose the best one manually.
python hallucinate.py --num_validation_image 8

# Step 3: Reconstruct textured 3D meshes, the result will be saved in data/examples/meshes
python reconstruct.py --test_folder data/examples --config recon/config.yaml --resume checkpoints/recon_model.pth