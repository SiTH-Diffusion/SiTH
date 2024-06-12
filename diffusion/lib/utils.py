"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import os
import torch
import numpy as np
from PIL import Image
import wandb

from diffusers import (
    UniPCMultistepScheduler,
)
from .pipeline import BackHallucinationPipeline

def tensor_to_np(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    _, w, h, _ = imgs[0].shape
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        pil_img = Image.fromarray((img[0] * 255).astype(np.uint8))
        grid.paste(pil_img, box=(i % cols * w, i // cols * h))
    return grid


def clip_encode_image_local(image, clip_image_encoder,
            refer_clip_proj=None, num_images_per_prompt=1, do_classifier_free_guidance=False): # clip local feature
    device = clip_image_encoder.device
    dtype = next(clip_image_encoder.parameters()).dtype

    image = image.to(device=device, dtype=dtype)
    last_hidden_states = clip_image_encoder(image).last_hidden_state
    last_hidden_states_norm = clip_image_encoder.vision_model.post_layernorm(last_hidden_states)
    if refer_clip_proj is None: # directly use clip pretrained projection layer
        image_embeddings = clip_image_encoder.visual_projection(last_hidden_states_norm)
    else:
        image_embeddings = refer_clip_proj(last_hidden_states_norm.to(dtype=dtype))
    # image_embeddings = image_embeddings.unsqueeze(1)

    # duplicate image embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
    image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

    if do_classifier_free_guidance:
        image = torch.zeros_like(image)
        image = image.to(device=device, dtype=dtype)
        last_hidden_states = clip_image_encoder(image).last_hidden_state
        last_hidden_states_norm = clip_image_encoder.vision_model.post_layernorm(last_hidden_states)
        if refer_clip_proj is None: # directly use clip pretrained projection layer
            negative_prompt_embeds = clip_image_encoder.visual_projection(last_hidden_states_norm)
        else:
            negative_prompt_embeds = refer_clip_proj(last_hidden_states_norm.to(dtype=dtype))
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        #negative_prompt_embeds = torch.zeros_like(image_embeddings)
            
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
         # to avoid doing two forward passes
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

    return image_embeddings.to(dtype=dtype)


def test_pipeline(logger, test_dataloader, vae, clip_image_encoder, unet, controlnet,
                           scheduler, refer_clip_proj, args, accelerator, weight_dtype, step, save_path):
    logger.info("Running test... ")

    controlnet = accelerator.unwrap_model(controlnet)
    unet = accelerator.unwrap_model(unet)
    refer_clip_proj = accelerator.unwrap_model(refer_clip_proj)

    pipeline = BackHallucinationPipeline(
        vae=vae,
        clip_image_encoder=clip_image_encoder,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        refer_clip_proj=refer_clip_proj,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    for i, data in enumerate(test_dataloader):
        images = []
        images.append(tensor_to_np(data['src_ori_image']))
        images.append(tensor_to_np(data['tgt_uv']))
        

        with torch.autocast("cuda"):
            ims = pipeline.forward(data, num_inference_steps=50, generator=generator,
                 guidance_scale=args.guidance_scale, controlnet_conditioning_scale=args.conditioning_scale,
                 num_images_per_prompt = args.num_gen_images
                )
        [images.append(image[None,...]) for image in ims]

        grid = image_grid(images, 1, args.num_gen_images + 2 )
        grid.save(os.path.join(save_path, f"%07d_%04d_output.png" % (step, i)))



def log_validation(logger, val_dataloader, vae, clip_image_encoder, unet, controlnet,
                           scheduler, refer_clip_proj, args, accelerator, weight_dtype, step):
    logger.info("Running validation... ")

    controlnet = accelerator.unwrap_model(controlnet)
    unet = accelerator.unwrap_model(unet)
    refer_clip_proj = accelerator.unwrap_model(refer_clip_proj)

    pipeline = BackHallucinationPipeline(
        vae=vae,
        clip_image_encoder=clip_image_encoder,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        refer_clip_proj=refer_clip_proj,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    image_logs = []

    for i, data in enumerate(val_dataloader):      

        if i >=args.num_validation_images:
            break

        with torch.autocast("cuda"):
            ims = pipeline.forward(data, num_inference_steps=50, generator=generator,
                 guidance_scale=args.guidance_scale, controlnet_conditioning_scale=args.conditioning_scale,
                 num_images_per_prompt = args.num_gen_images
                )

        image_logs.append(
            {
            "src_images" : tensor_to_np(data['src_image']),
            "uv_images":  tensor_to_np(data['tgt_uv']),
            "gt_images": tensor_to_np(data['tgt_image']),
            "images": ims})


    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                src_images = log["src_images"]
                uv_images = log["uv_images"]

                formatted_images = []
                formatted_images.append(np.asarray(uv_images))

                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(src_images, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                src_images = log["src_images"]
                uv_images = log["uv_images"]
                tgt_images = log["gt_images"]

                formatted_images.append(wandb.Image(tgt_images[0], caption="Ground truth images"))
                formatted_images.append(wandb.Image(uv_images[0], caption="Controlnet uv images"))
                formatted_images.append(wandb.Image(src_images[0], caption="src_images"))

                for i, image in enumerate(images):
                    image = wandb.Image(image, caption="Generated image %d" % i)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

        return image_logs
