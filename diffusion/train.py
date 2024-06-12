"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import logging
import math
import os
import random
import shutil
import torch
import torch.nn.functional as F
import torch.utils.checkpoint


from datetime import datetime, timedelta
from packaging import version
from tqdm.auto import tqdm

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, InitProcessGroupKwargs

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available

import transformers
from transformers import CLIPVisionModelWithProjection

from lib.config import parse_options, argparse_to_str
from lib.train_diffusion_dataset import TrainDiffDataset
from lib.test_diffusion_dataset import TestDiffDataset
from lib.ccprojection import CCProjection
from lib.utils import clip_encode_image_local, test_pipeline, log_validation

UV_TEMPLATE = '../data/smplx_uv.obj'
check_min_version("0.24.0")

logger = get_logger(__name__)

def main(args, args_str):


    project_dir =  os.path.join(
            args.output_dir,
            args.exp_name
        )

    logging_dir = os.path.join(
            project_dir,
            f'{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        )

    accelerator_project_config = ProjectConfiguration(project_dir=project_dir, logging_dir=logging_dir)
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=18000))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs]
    )

    ##################  Prepare logger and set verbosity ##################

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Info: \n{args_str}")

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    # Handle the repository creation
    if accelerator.is_main_process:
        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)

    ##################  Dataset preparation ##################

    train_dataset = TrainDiffDataset(args)

    train_dataloader =  torch.utils.data.DataLoader(dataset=train_dataset, 
                        batch_size=args.train_batch_size, 
                        shuffle=True, 
                        num_workers=args.dataloader_num_workers,
                        pin_memory=True)
    if args.validation:
        val_dataloader =  torch.utils.data.DataLoader(dataset=train_dataset, 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=0,
                        pin_memory=True)

    if args.test:
        if args.test_data_dir is None:
            logger.warn("No test data directory provided. Skipping test.")
        else:
            test_dataset = TestDiffDataset(args.test_data_dir, UV_TEMPLATE, size=args.resolution)

            test_dataloader =  torch.utils.data.DataLoader(dataset=test_dataset, 
                        batch_size=1, 
                        shuffle=False, 
                        num_workers=0,
                        pin_memory=True)

    ##################  Model preparation ##################

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet")
    unet_input_channel = unet.config.in_channels
    clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained('kxic/zero123-xl', subfolder="image_encoder")
    refer_clip_proj = CCProjection(clip_image_encoder)

    logger.info("Initializing controlnet weights from unet")

    controlnet = ControlNetModel.from_unet(unet, conditioning_channels=args.conditioning_channels)

    #if args.pretrained_model_name_or_path != 'kxic/zero123-xl':
    #    controlnet = ControlNetModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="controlnet")
    #    refer_clip_proj = CCProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="projection", clip_image_encoder=clip_image_encoder)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            i = len(weights) - 1
            while len(weights) > 0:
                weights.pop()
                model = models[i]
                if isinstance(model, UNet2DConditionModel):
                    sub_dir = "unet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                elif isinstance(model, ControlNetModel):
                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                elif isinstance(model, CCProjection):
                    sub_dir = "projection"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))
                else:
                    torch.save(model, os.path.join(output_dir, 'model.pth'))

                i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()
                if isinstance(model, UNet2DConditionModel):
                    load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                elif isinstance(model, ControlNetModel):
                    load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                elif isinstance(model, CCProjection):
                    load_model = CCProjection.from_pretrained(input_dir, subfolder="projection",clip_image_encoder=clip_image_encoder)
                else:
                    load_model = torch.load(os.path.join(input_dir, 'model.pth'))
                # load diffusers style into model
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)


    ## Freeze the weights of the vae, unet and text_encoder
    ## Only train the controlnet and the transformer blocks in the unet

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    clip_image_encoder.requires_grad_(False)

    controlnet.train()
    refer_clip_proj.train()
    for param_name, param in unet.named_parameters():
        if 'transformer_blocks' not in param_name:
            param.requires_grad_(False)
        else:
            param.requires_grad_(True)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")


    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()


    ##################  Optimizer creation ##################
    scale = 1.0
    if args.scale_lr:
        scale =  args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        

    params_to_optimize = []
    optimizer_class = torch.optim.AdamW

    params_to_optimize.append({'params': controlnet.parameters(),
                                'lr': args.lr_controlnet * scale,
                                })
    params_to_optimize.append({'params': refer_clip_proj.parameters(),
                                'lr': args.lr * scale,
                                'weight_decay': args.adam_weight_decay})    
    params_to_optimize.append({'params': unet.parameters(),
                                'lr': args.lr * scale,
                                'weight_decay': args.adam_weight_decay})        
    
    #params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    ##################  Scheduler creation ##################

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )


    ##################  Training preparation ##################

    # Prepare everything with our `accelerator`.
    unet, refer_clip_proj, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, refer_clip_proj, controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    clip_image_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)


    ##################  Trackers preparation ##################


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        init_kwargs = {"wandb": {'name': os.path.basename(logging_dir), 'dir': logging_dir}}
        accelerator.init_trackers(args.exp_name, config=tracker_config, init_kwargs=init_kwargs)

        for tracker in accelerator.trackers:
            if tracker.name == "wandb":
                tracker.run.watch(unet)
                tracker.run.watch(refer_clip_proj)
                tracker.run.watch(controlnet)



    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:

        path = args.resume_from_checkpoint

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = int(path.split("-")[-1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0


    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )



    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):

                # Convert images to latent space
                latents = vae.encode(batch["tgt_image"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                # encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                #controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                src_img = batch["src_image"]
                cond_img = batch["src_ori_image"]
                if args.drop_prob > 0:
                    p = random.random()
                    if p <= args.drop_prob: # dropout ref image
                        src_img = torch.zeros_like(src_img)
                        cond_img = torch.zeros_like(cond_img)

                img_latents = vae.encode(cond_img.to(dtype=weight_dtype)).latent_dist.mode()
                img_latents = img_latents * vae.config.scaling_factor

                encoder_hidden_states = clip_encode_image_local(src_img, clip_image_encoder, refer_clip_proj)

                if args.conditioning_channels == 8:
                    controlnet_image = torch.cat([batch['tgt_uv'], batch['view_cond'], batch['tgt_mask']], dim=1)
                elif args.conditioning_channels == 4:
                    controlnet_image = torch.cat([batch['tgt_uv'], batch['tgt_mask']], dim=1)
                else:
                    controlnet_image = batch['tgt_mask']

                if unet_input_channel == 8:
                    noisy_latents = torch.cat([noisy_latents, img_latents], dim=1)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    conditioning_scale=args.conditioning_scale,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                ).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(logging_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(logging_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        os.makedirs(os.path.join(logging_dir, f"checkpoint-{global_step}"), exist_ok=True)
                        save_path = os.path.join(logging_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation and global_step % args.validation_steps == 0:
                        torch.cuda.empty_cache()
                        _ = log_validation(
                            logger,
                            val_dataloader,
                            vae,
                            clip_image_encoder,
                            unet,
                            controlnet,
                            noise_scheduler,
                            refer_clip_proj,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )
                        torch.cuda.empty_cache()

                    if args.test and global_step % args.test_steps == 0:
                        save_path = os.path.join(logging_dir, "test_outputs")
                        os.makedirs(save_path, exist_ok=True)
                        torch.cuda.empty_cache()
                        _ = test_pipeline(
                            logger,
                            test_dataloader,
                            vae,
                            clip_image_encoder,
                            unet,
                            controlnet,
                            noise_scheduler,
                            refer_clip_proj,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            save_path
                        )
                        torch.cuda.empty_cache()



            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = accelerator.unwrap_model(controlnet)
        controlnet.save_pretrained(logging_dir)


    accelerator.end_training()


if __name__ == "__main__":
    parser = parse_options()
    args, args_str = argparse_to_str(parser)
    main(args, args_str)