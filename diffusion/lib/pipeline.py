# Copyright 2024 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from transformers import CLIPVisionModelWithProjection

from diffusers.loaders import FromSingleFileMixin, LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

class BackHallucinationPipeline(
    DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin, FromSingleFileMixin
):
    
    r"""
    Pipeline for text-to-image generation using Stable Diffusion with ControlNet guidance.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    In addition the pipeline inherits the following loading methods:
        - *Textual-Inversion*: [`loaders.TextualInversionLoaderMixin.load_textual_inversion`]

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        controlnet ([`ControlNetModel`] or `List[ControlNetModel]`):
            Provides additional conditioning to the unet during the denoising process. If you set multiple ControlNets
            as a list, the outputs from each ControlNet are added together to create one combined additional
            conditioning.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/runwayml/stable-diffusion-v1-5) for details.
        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae: AutoencoderKL,
        clip_image_encoder: CLIPVisionModelWithProjection,
        unet: UNet2DConditionModel,
        controlnet: Union[ControlNetModel, List[ControlNetModel], Tuple[ControlNetModel], MultiControlNetModel],
        scheduler: KarrasDiffusionSchedulers,
        refer_clip_proj: nn.Linear = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        if isinstance(controlnet, (list, tuple)):
            controlnet = MultiControlNetModel(controlnet)

        self.register_modules(
            vae=vae.cuda(),
            clip_image_encoder=clip_image_encoder.cuda(),
            unet=unet.cuda(),
            controlnet=controlnet.cuda(),
            refer_clip_proj=refer_clip_proj.cuda(),
            scheduler=scheduler,
            torch_dtype=torch_dtype,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)


    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        device = torch.device(f"cuda:{gpu_id}")

        hook = None
        for cpu_offloaded_model in [self.clip_image_encoder, self.unet, self.vae, self.refer_clip_proj]:
            _, hook = cpu_offload_with_hook(cpu_offloaded_model, device, prev_module_hook=hook)

        # control net hook has be manually offloaded as it alternates with unet
        cpu_offload_with_hook(self.controlnet, device)

        # We'll offload the last model manually.
        self.final_offload_hook = hook


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    def decode_latents(self, latents):
        warnings.warn(
            "The decode_latents method is deprecated and will be removed in a future version. Please"
            " use VaeImageProcessor instead",
            FutureWarning,
        )
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image
    

    def encode_images(self, images):

        # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_images
        # encode images
        latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = latents.to(dtype=self.torch_dtype)

        return latents


    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // self.vae_scale_factor, width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_img_latents(self, image, batch_size, dtype, device, do_classifier_free_guidance=False):

        image = image.to(device=self.device, dtype=dtype)
        init_latents = self.vae.encode(image).latent_dist.mode()
        init_latents = init_latents * self.vae.config.scaling_factor

        if batch_size > init_latents.shape[0]:
            # init_latents = init_latents.repeat(batch_size // init_latents.shape[0], 1, 1, 1)
            num_images_per_prompt = batch_size // init_latents.shape[0]
            # duplicate image latents for each generation per prompt, using mps friendly method
            bs_embed, emb_c, emb_h, emb_w = init_latents.shape
            init_latents = init_latents.unsqueeze(1)
            init_latents = init_latents.repeat(1, num_images_per_prompt, 1, 1, 1)
            init_latents = init_latents.view(bs_embed * num_images_per_prompt, emb_c, emb_h, emb_w)

        # init_latents = torch.cat([init_latents]*2) if do_classifier_free_guidance else init_latents   # follow zero123
        #init_latents = (
        #    torch.cat([torch.zeros_like(init_latents), init_latents]) if do_classifier_free_guidance else init_latents
        #)
        if do_classifier_free_guidance:
            zero_image = torch.zeros_like(image).to(device=self.device, dtype=dtype)
            zero_latents = self.vae.encode(zero_image).latent_dist.mode()
            zero_latents = zero_latents * self.vae.config.scaling_factor
            if batch_size > zero_latents.shape[0]:
                num_images_per_prompt = batch_size // zero_latents.shape[0]
                # duplicate image latents for each generation per prompt, using mps friendly method
                bs_embed, emb_c, emb_h, emb_w = zero_latents.shape
                zero_latents = zero_latents.unsqueeze(1)
                zero_latents = zero_latents.repeat(1, num_images_per_prompt, 1, 1, 1)
                zero_latents = zero_latents.view(bs_embed * num_images_per_prompt, emb_c, emb_h, emb_w)
            
            init_latents = torch.cat([zero_latents, init_latents])


        init_latents = init_latents.to(device=device, dtype=dtype)

        return init_latents



    def clip_encode_image_local(self, image, num_images_per_prompt=1, do_classifier_free_guidance=False): # clip local feature
        dtype = next(self.clip_image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(images=image, return_tensors="pt").pixel_values

        image = image.to(device=self.device, dtype=dtype)
        last_hidden_states = self.clip_image_encoder(image).last_hidden_state
        last_hidden_states_norm = self.clip_image_encoder.vision_model.post_layernorm(last_hidden_states)

        if self.refer_clip_proj is not None:
            image_embeddings = self.refer_clip_proj(last_hidden_states_norm.to(dtype=self.torch_dtype))
        else:
            image_embeddings = self.clip_image_encoder.visual_projection(last_hidden_states_norm)
        # image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            image = torch.zeros_like(image)
            image = image.to(device=self.device, dtype=dtype)
            last_hidden_states = self.clip_image_encoder(image).last_hidden_state
            last_hidden_states_norm = self.clip_image_encoder.vision_model.post_layernorm(last_hidden_states)
            if self.refer_clip_proj is not None: # directly use clip pretrained projection layer
                negative_prompt_embeds = self.refer_clip_proj(last_hidden_states_norm.to(dtype=self.torch_dtype))
            else:
                negative_prompt_embeds = self.clip_image_encoder.visual_projection(last_hidden_states_norm)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
            #negative_prompt_embeds = torch.zeros_like(image_embeddings)
            
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])

        return image_embeddings.to(dtype=self.torch_dtype)



    @torch.no_grad()
    def forward_enc_dec(self, inputs):
        image = inputs['tgt_imgs']
        latent = self.image_encoder(image)
        gen_img = self.image_decoder(latent)
        return gen_img

    def set_gradient(self, training=True):
        if training:
            self.unet.train()
            self.controlnet.train()
            self.refer_clip_proj.train()
        else:
            self.unet.eval()
            self.controlnet.eval()
            self.refer_clip_proj.eval()

    
    @torch.no_grad()
    def forward(self, inputs, generator, num_inference_steps: int = 50, guidance_scale: float = 7.5,
                  num_images_per_prompt: int = 1, controlnet_conditioning_scale= 1.0,
                  guess_mode: bool = False, control_guidance_start = 0.0, control_guidance_end= 1.0):
        
        if not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(self.controlnet.nets) if isinstance(self.controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        src_img = inputs['src_image'].to(device=device, dtype=self.torch_dtype)
        
        # prepare source embedding
        refer_latents = self.clip_encode_image_local(src_img,
                num_images_per_prompt = num_images_per_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance).to(dtype=self.torch_dtype)

        # prepare controlnet input
        tgt_uv = inputs['tgt_uv'].to(device=device, dtype=self.torch_dtype)
        view_cond = inputs['view_cond'].to(device=device, dtype=self.torch_dtype)
        tgt_mask = inputs['tgt_mask'].to(device=device, dtype=self.torch_dtype)

        batch_size, _, height, width = inputs['tgt_uv'].shape

        if self.controlnet.config.conditioning_channels == 4:
            cond_input = torch.cat([tgt_uv, tgt_mask], dim=1)
        elif self.controlnet.config.conditioning_channels == 8:
            cond_input = torch.cat([tgt_uv, view_cond, tgt_mask], dim=1)
        else:
            cond_input = tgt_mask

        cond_input = cond_input.repeat_interleave(num_images_per_prompt, dim=0)
        if do_classifier_free_guidance and not guess_mode:
            cond_input = torch.cat([cond_input] * 2)

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 6. Prepare latent variables
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            refer_latents.dtype,
            device,
            generator,
        )


        cond_img = inputs['src_ori_image']

        img_latents = self.prepare_img_latents(
            cond_img,
            batch_size * num_images_per_prompt,
            refer_latents.dtype,
            device,
            do_classifier_free_guidance = do_classifier_free_guidance,
        )

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(self.controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                if self.unet.config.in_channels == 8:
                    latent_model_input = torch.cat([latent_model_input, img_latents], dim=1)

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = refer_latents.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = refer_latents

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=cond_input,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=refer_latents,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        # Post-processing
        gen_img = self.decode_latents(latents)
        return gen_img
