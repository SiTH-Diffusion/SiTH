"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""

import argparse
import os
import numpy as np
import torch
import torch.utils.checkpoint
from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from accelerate.utils import  set_seed
from transformers import CLIPVisionModelWithProjection

from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

from diffusion.lib.test_diffusion_dataset import TestDiffDataset
from diffusion.lib.pipeline import BackHallucinationPipeline
from diffusion.lib.ccprojection import CCProjection
from diffusion.lib.utils import tensor_to_np, image_grid

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.24.0")
UV_TEMPLATE = 'data/smplx_uv.obj'

def main(args):
    
    os.makedirs(args.output_path, exist_ok=True)
    logging_dir = os.path.join(args.output_path, 'all_images')
    os.makedirs(logging_dir, exist_ok=True)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weight_dtype = torch.float32

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)
    else:
        generator = None

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="image_encoder")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="controlnet")
    refer_clip_proj = CCProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="projection", clip_image_encoder=clip_image_encoder)
    
    # Freeze the model
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)
    controlnet.requires_grad_(False)
    refer_clip_proj.requires_grad_(False)

    # Load the dataset
    val_dataset = TestDiffDataset(args.input_path, UV_TEMPLATE, size=args.resolution)

    val_dataloader =  torch.utils.data.DataLoader(dataset=val_dataset, 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=0,
                        pin_memory=True)

    pipeline = BackHallucinationPipeline(
        vae=vae,
        clip_image_encoder=clip_image_encoder,
        unet=unet,
        controlnet=controlnet,
        scheduler=noise_scheduler,
        refer_clip_proj=refer_clip_proj,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                print(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipeline.enable_xformers_memory_efficient_attention()

        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    for i, data in enumerate(tqdm(val_dataloader)):
        images = []
        images.append(tensor_to_np(data['src_ori_image']))
        images.append(tensor_to_np(data['tgt_uv']))
        fname = data['filename'][0]

        with torch.autocast("cuda"):
                im = pipeline.forward(data, num_inference_steps=args.num_inference_steps, generator=generator,
                 guidance_scale=args.guidance_scale, controlnet_conditioning_scale=args.conditioning_scale,
                 num_images_per_prompt = args.num_validation_images
                )
        for j in range(args.num_validation_images):

            pil_img = Image.fromarray((im[j] * 255).astype(np.uint8))
            pil_img.save(os.path.join(logging_dir,  f"%s_%03d.png" % (fname, j)))
            if j == 0:
                pil_img.save(os.path.join(args.output_path,  f"%s_%03d.png" % (fname, j)))

            images.append(im[j:j+1])

        grid = image_grid(images, 1, args.num_validation_images +2 )
        grid.save(os.path.join(logging_dir, f"{fname}_all.png"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        default="./data/examples",
        help=(
            "The path to the dataset. The directory should contain a images folder and a smplx meshes folder."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="./data/examples/back_images",
        help=(
            "The output path for the generated images. The generated images will be saved in this path."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default='hohs/SiTH_diffusion',
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="classifier free guidence scale"
    )
    parser.add_argument(
        "--conditioning_scale",
        type=float,
        default=1.0,
        help="Controlnet conditioning scale"
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of inference steps",
    )

    args = parser.parse_args()


    main(args)
