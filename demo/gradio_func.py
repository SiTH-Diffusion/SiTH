"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""

import os
import cv2
import math
import json
import pickle
import nvdiffrast
import torch
import torch.nn.functional as F
import numpy as np
import kaolin as kal
import gradio as gr


from torchvision import transforms
from PIL import Image, ImageColor
from smplx import SMPLX
from torchvision.transforms.functional import normalize

from transformers import CLIPVisionModelWithProjection
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetPipeline
)
from recon.models.evaluator import Evaluator
from diffusion.lib.pipeline import BackHallucinationPipeline
from diffusion.lib.ccprojection import CCProjection
from fitting.utils.kps import draw_openpose_keypoints, vis_meshes

from .briarmbg import BriaRMBG
from .load_obj import load_obj

####################################################################################
# Define global variables
####################################################################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)

####################################################################################
# Load template files
####################################################################################

UV_TEMPLATE = 'data/smplx_uv.obj'
SMPL_PATH = 'data/body_models/smplx'
WATERTIGHT_TEMPLATE = 'data/smplx_watertight.pkl'
CANONICAL_TEMPLATE = 'data/smplx_canonical.obj'
EVALUATOR_CONFIG = 'data/gradio_config.pkl'

can_V, _ = load_obj(CANONICAL_TEMPLATE)
_, _, texv, texf, _ = load_obj(UV_TEMPLATE, load_materials=True)

with open(WATERTIGHT_TEMPLATE, 'rb') as f:
    watertight = pickle.load(f)

####################################################################################
# Define rendering camera
####################################################################################

look_at = torch.zeros( (2, 3), dtype=torch.float32, device=device)
camera_up_direction = torch.tensor( [[0, 1, 0]], dtype=torch.float32, device=device)
camera_position = torch.tensor(  [[0, 0, 3],
                                 [0, 0, -3]] , dtype=torch.float32, device=device)


camera = kal.render.camera.Camera.from_args(eye=camera_position,
                                         at=look_at,
                                         up=camera_up_direction,
                                         width=512, height=512,
                                         near=-512, far=512,
                                        fov_distance=1.0, device=device)

####################################################################################
# Define the image transformations
####################################################################################

transform= transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

transform_rgba= transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5,0.0], std=[0.5,0.5,0.5,1.0])
        ])
transform_clip = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073, 0.0],
                                 [0.26862954, 0.26130258, 0.27577711, 1.0]),
        ])
up_scale = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
up_scale_rgba = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5,0.0], std=[0.5,0.5,0.5,1.0])
        ])

####################################################################################
# Initialize 2D ControlNet pipeline
####################################################################################
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "hohs/hohs_mix",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# This command loads the individual model components on GPU on-demand. So, we don't
# need to explicitly call pipe.to("cuda").
pipe.enable_model_cpu_offload()

# xformers
#pipe.enable_xformers_memory_efficient_attention()

####################################################################################
# Initialize 2D ControlNet pipeline
####################################################################################

model_name = 'hohs/SiTH-diffusion-2000'
noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
clip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(model_name, subfolder="image_encoder")
unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet")
controlnet_ours = ControlNetModel.from_pretrained(model_name, subfolder="controlnet")
refer_clip_proj = CCProjection.from_pretrained(model_name, subfolder="projection", clip_image_encoder=clip_image_encoder)


pipeline = BackHallucinationPipeline(
        vae=vae,
        clip_image_encoder=clip_image_encoder,
        unet=unet,
        controlnet=controlnet_ours,
        scheduler=noise_scheduler,
        refer_clip_proj=refer_clip_proj,
        torch_dtype=torch.float32,
    )

pipeline.to('cpu')

####################################################################################
# Initialize reconstruction model
####################################################################################

with open(EVALUATOR_CONFIG, 'rb') as f:
    config = pickle.load(f)
config.grid_size = 400

evaluator = Evaluator(config, watertight, can_V)
evaluator.to('cpu')

####################################################################################
# Initialize reconstruction model
####################################################################################

bgnet = BriaRMBG.from_pretrained("briaai/RMBG-1.4")
bgnet.to('cpu')

####################################################################################
# Function definitions
####################################################################################

def get_select_index(images_list, evt: gr.SelectData):

    return images_list[evt.index]

def load_smpl_json(json_path, scale=0.9, offset_x = 0.0, offset_y = 0.1):
    smpl_data = json.load(open(json_path))
    mapper = json.load(open('data/smplx_openpose25.json'))['smplx_idxs']

    param_betas = torch.tensor(smpl_data['betas'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_poses = torch.tensor(smpl_data['body_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_left_hand_pose = torch.tensor(smpl_data['left_hand_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_right_hand_pose = torch.tensor(smpl_data['right_hand_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
            
    param_expression = torch.tensor(smpl_data['expression'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_jaw_pose = torch.tensor(smpl_data['jaw_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_leye_pose = torch.tensor(smpl_data['leye_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_reye_pose = torch.tensor(smpl_data['reye_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    
    body_model = SMPLX(model_path=SMPL_PATH, gender='male', use_pca=True, num_pca_comps=12,
                        flat_hand_mean=True, use_face_contour=True).to(device)
    
    J_0 = body_model(body_pose = param_poses, betas=param_betas).joints.contiguous().detach()
    output = body_model(betas=param_betas,
                                   body_pose=param_poses,
                                   transl=-J_0[:,0,:],
                                   left_hand_pose=param_left_hand_pose,
                                   right_hand_pose=param_right_hand_pose,
                                   expression=param_expression,
                                   jaw_pose=param_jaw_pose,
                                   leye_pose=param_leye_pose,
                                   reye_pose=param_reye_pose,
                                   )

    V = output.vertices.detach().to(device) * scale + torch.tensor([offset_x, offset_y, 0], device=device)
    F = torch.tensor(body_model.faces.astype(int)).to(device)
    joint_3d = output.joints.detach().to(device)[:, mapper] * scale + torch.tensor([offset_x, offset_y, 0], device=device)

    return V, F, joint_3d

def render_uv_map(camera, V, F, texv, texf, size):

    vertices_camera = camera.extrinsics.transform(V)
    face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(
                        vertices_camera, F)
    face_normals_z = kal.ops.mesh.face_normals(face_vertices_camera,unit=True)[..., -1:].contiguous()
    proj = camera.projection_matrix()[0:1]
    homogeneous_vecs = kal.render.camera.up_to_homogeneous(
            vertices_camera
        )[..., None]

    vertices_clip = (proj @ homogeneous_vecs).squeeze(-1)
    rast = nvdiffrast.torch.rasterize(
            glctx, vertices_clip, F.int(),
            (size, size), grad_db=False
        )
    rast0 = torch.flip(rast[0], dims=(1,))
    hard_mask = rast0[:, :, :, -1:] != 0

    uv_map = nvdiffrast.torch.interpolate(
            texv.cuda(), rast0, texf[...,:3].int().cuda()
    )[0] % 1.
    return uv_map

def render_visiblity(camera, V, F, size=(512, 512)):

    vertices_camera = camera.extrinsics.transform(V)
    face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(
                            vertices_camera, F)
    face_normals_z = kal.ops.mesh.face_normals(face_vertices_camera,unit=True)[..., -1:].contiguous()
    proj = camera.projection_matrix()[0:1]
    homogeneous_vecs = kal.render.camera.up_to_homogeneous(
            vertices_camera
        )[..., None]
    vertices_clip = (proj @ homogeneous_vecs).squeeze(-1)
    rast = nvdiffrast.torch.rasterize(
            glctx, vertices_clip, F.int(),
            size, grad_db=False
        )
    rast0 = torch.flip(rast[0], dims=(1,))
    face_idx = (rast0[..., -1:].long() - 1).contiguous()
    # assign visibility to 1 if face_idx >= 0
    vv = []
    for i in range(rast0.shape[0]):
        vis = torch.zeros((F.shape[0],), dtype=torch.bool, device=device)
        for f in range(F.shape[0]):
            vis[f] = 1 if torch.any(face_idx[i] == f) else 0
        vv.append(vis)

    front_vis = vv[0].bool()
    back_vis = vv[1].bool()
    vis_class = torch.zeros((F.shape[0], 1), dtype=torch.float32)
    vis_class[front_vis] = 1.0
    vis_class[back_vis] = -1.0

    return vis_class

def extract_openpose_keypoints(json_path, scale, offset_x, offset_y):

    V, F, joint_3d = load_smpl_json(json_path, scale, offset_x, offset_y)

    uv = render_uv_map(camera[1:], V.to(device), F.to(device), texv, texf, 512)

    tgt_uv = torch.zeros((512, 512, 3))
    tgt_uv[..., :2] = uv[0].cpu()
    tgt_uv = tgt_uv.permute(2,0,1) * 2. - 1


    water_tight_F = watertight['smpl_F'].to(device)

    vis = render_visiblity(camera, V.to(device), water_tight_F)


    V_2d = V[0,:,:2].detach().cpu().numpy() * np.array([[512 // 2, -512 // 2]]) + np.array([512 // 2, 512// 2])

    canvas = np.zeros((512, 512, 3), dtype=np.uint8)

    kps = joint_3d[0]
    kps [..., 2] = 1.0

    kp_img = draw_openpose_keypoints(kps, canvas, height=512, width=512, skeleton='COCO19')

    return Image.fromarray(kp_img), Image.fromarray(kp_img), V_2d, V, tgt_uv, vis, json_path


def get_pose_from_example(example_img):
    filename = example_img.split('/')[-1]
    json_path = os.path.join('data', 'gradio', filename.split('.')[0] + '.json')
    
    return extract_openpose_keypoints (json_path, 1.0, 0.0, 0.0)


def generate_images(control_image, pos_prompt, neg_prompt, scheduler_name, num_steps, num_images, seed, cfg_scale, cond_scale):
    generator = torch.manual_seed(seed)

    if scheduler_name == 'DPM++ 2M':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == 'DPM++ 2M Karras':
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == 'DPM2':
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == 'DPM2 Karras':
        pipe.scheduler = KDPM2DiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == 'DPM2 a':
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == 'DPM2 a Karras':
        pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == 'Euler':
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == 'Euler Ancestral':
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    if control_image is None:
        raise gr.Error("Please provide an image")
    try:
        output = pipe(
            pos_prompt,
            image=[control_image],
            generator=generator,
            negative_prompt=neg_prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps,
            guidance_scale=float(cfg_scale),
            controlnet_conditioning_scale=float(cond_scale)
        )
        all_outputs = []
        for im in output.images:
            all_outputs.append(im)


    except Exception as e:
        raise gr.Error(str(e))
    

    torch.cuda.empty_cache()
    return all_outputs, all_outputs


def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


def rm_background(image, V_2d):
    bgnet.to(device)

    img_ori = image.copy()
    # prepare input
    w,h = image.size
    vis_img = np.array(image)

    new_image = resize_image(image)
    im_np = np.array(new_image)
    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
    im_tensor = torch.unsqueeze(im_tensor,0)
    im_tensor = torch.divide(im_tensor,255.0)
    im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
    if torch.cuda.is_available():
        im_tensor=im_tensor.cuda()

    #inference
    result=bgnet(im_tensor)
    # post process
    result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)    
    # image to pil
    im_array = (result*255).squeeze().cpu().data.numpy().astype(np.uint8)

    rgba = np.concatenate([vis_img, np.expand_dims(im_array, axis=-1)], axis=-1)

    overlap_img = vis_meshes(vis_img, V_2d, alpha=0.8, radius=2, color=(0, 0, 255))

    pil_im = Image.fromarray(rgba, 'RGBA')
    bgnet.to('cpu')

    return img_ori, pil_im, pil_im, overlap_img


def select_gen_images(images_list, V_2d, evt: gr.SelectData):

    return rm_background(images_list[evt.index], V_2d)


def hallucinate(rgba_image, tgt_uv, scheduler_name, num_steps, num_images, seed, cfg_scale, cond_scale):

    generator = torch.manual_seed(seed) 

    if scheduler_name == 'DPM++ 2M':
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_name == 'DPM++ 2M Karras':
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == 'DPM2':
        pipeline.scheduler = KDPM2DiscreteScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_name == 'DPM2 Karras':
        pipeline.scheduler = KDPM2DiscreteScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == 'DPM2 a':
        pipeline.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_name == 'DPM2 a Karras':
        pipeline.scheduler = KDPM2AncestralDiscreteScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True)
    elif scheduler_name == 'Euler':
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    elif scheduler_name == 'Euler Ancestral':
        pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
    else:
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)

    pipeline.to(device)
    src_rgba = transform_clip(rgba_image)
    src_mask = src_rgba[-1:,...]
    src_rgb = src_rgba[:-1,...]
    src_clip_image = src_rgb * src_mask


    src_rgba = transform_rgba(rgba_image)
    src_mask = src_rgba[-1:,...]
    src_rgb = src_rgba[:-1,...]
    src_image = src_rgb * src_mask

    view_cond = torch.stack(
            [   torch.tensor(0.0),
                torch.sin(torch.tensor(math.pi)),
                torch.cos(torch.tensor(math.pi)),
                torch.tensor(0.0)] ).view(-1,1,1).repeat(1, 512, 512)

    data = {
        'src_ori_image': src_image[None, ...].to(device),
        'src_image': src_clip_image[None, ...].to(device),
        'tgt_uv' : tgt_uv[None, ...].to(device),
        'view_cond' : view_cond[None, ...].to(device),
        'tgt_mask': torch.flip(src_mask, [2])[None, ...].to(device),

    }
    
    if rgba_image is None or tgt_uv is None:
        raise gr.Error("Please provide an image")
    try:
        output = pipeline.forward(data, num_inference_steps=num_steps, generator=generator, 
                 guidance_scale=cfg_scale, controlnet_conditioning_scale=cond_scale, num_images_per_prompt=num_images,
                )

        back_images = []
        for i in range(output.shape[0]):
            pil_img = Image.fromarray((output[i] * 255).astype(np.uint8))
            back_images.append(pil_img)
    
    except Exception as e:
        raise gr.Error(str(e))

    pipeline.to('cpu')
    torch.cuda.empty_cache()

    return back_images, back_images



def add_background(image, mask, color=(0.0, 0.0, 0.0)):
    # Random background
    bg_color = ( torch.tensor(color).float() / 255.) - 0.5
    bg = torch.ones_like(image) * bg_color.view(3,1,1)
    _mask = (mask<0.5).expand_as(image)
    image[_mask] = bg[_mask]

    return image
    

def erode_mask(mask, kernal=(5,5), iter=1):
    mask = torch.from_numpy(cv2.erode(mask[0].numpy(), np.ones(kernal, np.uint8), iterations=iter)).float().unsqueeze(0)
    return mask

def reconstruct(select_front, select_back, V_3d, vis, bg_color, iters):

    evaluator.to(device)
    
    front_img = up_scale_rgba(select_front)
    front_rgb = front_img[:3,...]
    mask = front_img[3:,...]
    color = ImageColor.getcolor(bg_color, "RGB") 

    mask = erode_mask(mask, kernal=(5,5), iter=iters)
    front_rgb = add_background(front_rgb, mask, color = color)

    back_img = torch.flip(up_scale(select_back), [2])
    back_img = add_background(back_img, mask, color = color)

    front_pil = Image.fromarray(((front_rgb.cpu().numpy().transpose(1,2,0) * 0.5 + 0.5) * 255).astype(np.uint8)).convert('RGB')
    back_pil = Image.fromarray(((back_img.cpu().numpy().transpose(1,2,0) * 0.5 + 0.5) * 255).astype(np.uint8)).convert('RGB')

    data = {
        'fname': 'test',
        'smpl_v': V_3d.to(device),
        'vis_class': vis[None, ...].to(device),
        'front_rgb_img': front_rgb[None, ...].to(device),
        'back_rgb_img': back_img[None, ...].to(device),
        'mask': mask
        }
    save_path = 'data/examples'
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        mesh = evaluator.test_reconstruction(data, save_path, True, chunk_size=1e5, flip=True)
    
    evaluator.to('cpu')
    torch.cuda.empty_cache()

    return mesh, front_pil, back_pil
