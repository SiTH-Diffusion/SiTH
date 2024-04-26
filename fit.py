"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""

import os
import cv2
import argparse
import trimesh
import json
import numpy as np
from smplx import SMPLX
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
import kaolin as kal

from fitting.models.smpler_x_model import Model
from fitting.utils.conversion import rotation_matrix_to_angle_axis, batch_rodrigues
from fitting.utils.kps import draw_openpose_keypoints, load_openpose_json, vis_meshes

# Define the data paths
CKPT_PATH = 'checkpoints/save_smplerx.pth'
SMPL_PATH = 'data/body_models'
JOINT_MAP = 'data/smplx_openpose25.json'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
mapper = json.load(open(JOINT_MAP))['smplx_idxs']
transform = transforms.ToTensor()

# COCO25 keypoints for optimization
body_idx = [1, 2, 5, 8, 9, 12]
leg_idx = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
lhand_idx = [5, 6, 7, 29, 33, 37, 41, 45]
rhand_idx = [2, 3, 4, 50, 54, 58, 62, 66]
face_idx = [0, 15, 16, 17, 18, 107, 110, 113, 116]

##################################################
# Optimization Hyperparameters
# You can adjust these hyperparameters for better fitting results
# The default values are set to the values used in the paper
OPT_ITER = 200

BODY_LOSS_WEIGHT = 20.0
HAND_LOSS_WEIGHT = 10.0
FACE_LOSS_WEIGHT = 20.0
LEG_LOSS_WEIGHT = 10.0
MASK_LOSS_WEIGHT = 1.0

LR = 1e-3
LR_ORIENT = 1e-3
LR_BETAS = 1e-3
LR_POSE = 1e-4

# If --opt_pos is activated, only the selected joints are optimized
# Note that the indices are based on SMPL-X joint order
OPT_JOINT_IDX = [14, 15, 16, 17, 18, 19, 20]
##################################################

def add_margin(pil_img, margin, crop, color):
    '''Add margin to the image and crop the image to the model input size'''

    width, height = pil_img.size
    new_width = width - 2 * crop
    new_height = height + 2 * margin
    result = Image.new(pil_img.mode, (width, new_height), color)
    result.paste(pil_img, (0, margin))

    # Resize the image to model input size
    im_resized = result.resize((384, 512),
                resample= Image.Resampling.LANCZOS.LANCZOS,
                box=(crop, 0, crop + new_width, new_height))

    return im_resized

def main(args):

    os.makedirs(args.output_path, exist_ok=True)
    if args.debug:
        debug_folder = os.path.join(args.output_path, 'debug')
        os.makedirs(debug_folder, exist_ok=True)

    # Load the model
    model = Model(CKPT_PATH)
    model.eval()

    # Initialize orthographic camera for fitting
    camera_position = torch.tensor( [0, 0, 3], dtype=torch.float32, device=device).unsqueeze(0)
    look_at = torch.zeros( (1, 3), dtype=torch.float32, device=device)
    camera_up_direction = torch.tensor( [0, 1, 0], dtype=torch.float32, device=device).unsqueeze(0)
    camera = kal.render.camera.Camera.from_args(eye=camera_position,
                                         at=look_at,
                                         up=camera_up_direction,
                                         width=args.size, height=args.size,
                                         near=-512, far=512,
                                        fov_distance=1.0, device='cuda')

    # Load the image and openpose keypoints
    img_list = [ os.path.join(args.input_path, x) for x in sorted(os.listdir(args.input_path)) if x.endswith(('.png'))]
    json_list = [ os.path.join(args.input_path, x) for x in sorted(os.listdir(args.input_path)) if x.endswith('.json')]

    for i, (input, pose) in enumerate(zip(img_list, json_list)):
        file_name = input.split('/')[-1].split('.')[0]
        img = Image.open(input)

        assert img.width == args.size and img.height == args.size

        # Pad the image to fit the model input size
        im_new = add_margin(img, args.size // 8, args.size // 32, (0,0,0,0))
        if args.debug:
            im_new.save(os.path.join(debug_folder, file_name + '_padded.png'))

        rgbd = transform(im_new)
        rgb = rgbd.cuda()[None, :3, :, :]
 
        inputs = {'img': rgb}

        # Model inference
        with torch.no_grad():
            out = model(inputs)


        # Load GT mask for fitting
        ori = transform(img)
        gt_mask = ori.cuda()[None, 3, :, :]

        # Load GT keypoints for fitting
        keypoints = load_openpose_json(pose)
        kps = torch.tensor(keypoints).detach().to(device)

        # Normalize the keypoints to [-1, 1] and flip the y-axis
        kps[:, 0] = ((kps[:, 0] / args.size) - 0.5 ) * 2.0
        kps[:, 1] = ((kps[:, 1] / args.size) - 0.5 ) * -2.0
        
        if args.debug:
            np_img = cv2.resize(cv2.imread(input), (args.size, args.size))
            np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
            kp_img = draw_openpose_keypoints(kps, np_img, height=args.size, width=args.size)
            kp_img = cv2.cvtColor(kp_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(debug_folder, f'{file_name}_kps.png'), kp_img)

        # Extract the COCO25 keypoints for optimization
        body_kps = kps[body_idx, :2]
        conf_body_kps = kps[body_idx, 2]

        leg_kps = kps[leg_idx, :2]
        conf_leg_kps = kps[leg_idx, 2]

        lhand_kps = kps[lhand_idx, :2]
        conf_lhand_kps = kps[lhand_idx, 2]

        rhand_kps = kps[rhand_idx, :2]
        conf_rhand_kps = kps[rhand_idx, 2]

        face_kps = kps[face_idx, :2]
        conf_face_kps = kps[face_idx, 2]

        # Initialize the SMPL-X parameters
        param_betas = out['smplx_shape'].clone().detach().reshape(1,-1).contiguous() # [1, 10]
        param_poses = out['smplx_body_pose'].clone().detach().reshape(1,-1).contiguous() # [1, 63]
        param_left_hand_pose = out['smplx_lhand_pose'].clone().detach().reshape(1,-1).contiguous() # [1, 15]
        param_right_hand_pose = out['smplx_rhand_pose'].clone().detach().reshape(1,-1).contiguous() # [1, 15]
        param_expression = out['smplx_expr'].clone().detach().reshape(1,-1).contiguous() # [1, 10]
        param_jaw_pose = out['smplx_jaw_pose'].clone().detach().reshape(1,-1).contiguous() # [1, 3]
        
        orient_angle = batch_rodrigues(out['smplx_root_pose']) # [1, 3]
        p = torch.tensor(np.pi)
        c, s = torch.cos(p), torch.sin(p)

        # Rotation matrix for 180-degree rotation around z-axis
        Rx = torch.tensor([[1, 0, 0],
                          [0, c, s],
                          [0, -s, c]]).to(device) 
    
        aa = Rx.T @ orient_angle 

        param_global_orient = rotation_matrix_to_angle_axis(aa).squeeze().detach().cpu().data

        # We used SMPL-X male model for fitting and mesh reconstruction
        body_model = SMPLX(model_path=os.path.join(SMPL_PATH, 'smplx'), gender='male', use_pca=False,
                        flat_hand_mean=False, use_face_contour=True).to(device)
        
        # cooridnates of the pelvis joint, we set this to the origin
        J_0 = body_model(body_pose = param_poses, betas=param_betas).joints.contiguous().detach()

        opt_offset_x = torch.zeros(1, device=device, requires_grad=True)
        opt_offset_y = torch.zeros(1, device=device, requires_grad=True)
        opt_scale = torch.ones(1, device=device, requires_grad=True)

        opt_betas = param_betas.requires_grad_(True)

        opt_pose_id = OPT_JOINT_IDX
        opt_pose = param_poses.reshape(1, -1, 3)[0, opt_pose_id].requires_grad_(True) # only selected joints are optimized

        opt_global_orient = torch.tensor([[param_global_orient[0],
                                              param_global_orient[1],
                                              param_global_orient[2]]],
                                              device=device, requires_grad=True)

        opt_params = []
        opt_params.extend([
            {
                "params": [opt_offset_x, opt_offset_y],
                'lr': LR
            },
            {
                "params": opt_scale,
                'lr': LR
            }
        ])
        if args.opt_orient:
            opt_params.append({
                "params": opt_global_orient,
                'lr': LR_ORIENT
            })
        if args.opt_betas:
            opt_params.append({
                "params": opt_betas,
                'lr': LR_BETAS
            })
        if args.opt_pose:
            opt_params.append({
                "params": opt_pose,
                'lr': LR_POSE
            })
            
        optimizer_smpl = torch.optim.Adam(opt_params, betas=(0.9, 0.999), amsgrad=True)
        
        scheduler_smpl = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_smpl,
            mode="min",
            factor=0.5,
            verbose=0,
            min_lr=1e-5
        )
        
        loop_smpl = tqdm(range(OPT_ITER))

        for i in loop_smpl:

            optimizer_smpl.zero_grad()
            full_pose = param_poses.clone().view(1, -1, 3)
            full_pose[0, opt_pose_id] = opt_pose

            transl = -J_0[:,0,:] + torch.cat([opt_offset_x, opt_offset_y, torch.zeros(1, device=device)], dim=0)
            output = body_model(global_orient=opt_global_orient,
                        betas=opt_betas,
                        body_pose=full_pose.view(1, -1),
                        transl=transl,
                        left_hand_pose=param_left_hand_pose,
                        right_hand_pose=param_right_hand_pose,
                        expression=param_expression,
                        jaw_pose=param_jaw_pose
                        )
            V = output.vertices * opt_scale
            F = torch.tensor(body_model.faces.astype(int)).to(device)
            joint_3d = output.joints[:, mapper] * opt_scale

            body_loss = (torch.norm(body_kps - joint_3d[0, body_idx, :2], dim=1) * conf_body_kps).mean(dim=0)

            hand_loss = (torch.norm(lhand_kps - joint_3d[0, lhand_idx, :2], dim=1) * conf_lhand_kps).mean(dim=0) + \
                        (torch.norm(rhand_kps - joint_3d[0, rhand_idx, :2], dim=1) * conf_rhand_kps).mean(dim=0) 
            
            face_loss = (torch.norm(face_kps - joint_3d[0, face_idx, :2], dim=1) * conf_face_kps).mean(dim=0)

            leg_loss = (torch.norm(leg_kps - joint_3d[0, leg_idx, :2], dim=1) * conf_leg_kps).mean(dim=0)

            # Render the mesh and compute the mask loss
            vertices_camera = camera.extrinsics.transform(V)
            face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(
                            vertices_camera, F)
            face_normals_z = kal.ops.mesh.face_normals(face_vertices_camera, unit=True)[..., -1:].contiguous()
            proj = camera.projection_matrix()[0:1]
            homogeneous_vecs = kal.render.camera.up_to_homogeneous(
            vertices_camera
            )[..., None]
            vertices_clip = (proj @ homogeneous_vecs).squeeze(-1)
            face_vertices_image = kal.ops.mesh.index_vertices_by_faces(
                            vertices_clip[...,0:2], F)
            face_attributes = [
                V[0][F].unsqueeze(0),
                ]  
            _, soft_mask, _ = kal.render.mesh.dibr_rasterization(
                    args.size, args.size, face_vertices_camera[:, :, :, -1],
                    face_vertices_image, face_attributes, face_normals_z,
                    sigmainv=20000, boxlen=0.05, knum=30, rast_backend='cuda')
        
            mask_loss = kal.metrics.render.mask_iou(soft_mask,gt_mask)
            reg_loss = torch.abs(opt_offset_x).item() + torch.abs(opt_offset_y).item() + torch.abs(opt_scale - 1.0).item()

            # Weighted sum of the losses
            smpl_loss = 0.0
            smpl_loss += body_loss * BODY_LOSS_WEIGHT + \
                         hand_loss * HAND_LOSS_WEIGHT + \
                         face_loss * FACE_LOSS_WEIGHT + \
                         leg_loss * LEG_LOSS_WEIGHT + \
                         mask_loss * MASK_LOSS_WEIGHT + \
                         reg_loss
            
            pbar_desc = "Body Fitting -- "
            pbar_desc += f"scale: {opt_scale.item():.3f} | x: {opt_offset_x.item():.3f} | y: {opt_offset_y.item():.3f} | "
            pbar_desc += f"Body: {body_loss:.3f} | "
            pbar_desc += f"Hand: {hand_loss:.3f} | "
            pbar_desc += f"Face: {face_loss:.3f} | "
            pbar_desc += f"Leg: {leg_loss:.3f} | "
            pbar_desc += f"Mask: {mask_loss:.3f} | "
            pbar_desc += f"Total: {smpl_loss:.3f}"
            
            loop_smpl.set_description(pbar_desc)

            smpl_loss.backward()
            optimizer_smpl.step()
            scheduler_smpl.step(smpl_loss)

        # Finish the optimization, save the results
        d = trimesh.Trimesh(vertices=V[0].detach().cpu().numpy(),faces=F.cpu().numpy())
        d.export(os.path.join(args.output_path, f'{file_name}_smplx.obj'))

        if args.debug:

            json_dict = {}
            json_dict['global_orient'] = opt_global_orient.reshape(-1).detach().cpu().numpy().tolist()
            json_dict['body_pose'] = full_pose.reshape(-1).detach().cpu().numpy().tolist()
            json_dict['betas'] = opt_betas.reshape(-1).detach().cpu().numpy().tolist()
            json_dict['left_hand_pose'] = param_left_hand_pose.reshape(-1).cpu().numpy().tolist()
            json_dict['right_hand_pose'] = param_right_hand_pose.reshape(-1).cpu().numpy().tolist()
            json_dict['jaw_pose'] = param_jaw_pose.reshape(-1).cpu().numpy().tolist()
            json_dict['expression'] = param_expression.reshape(-1).cpu().numpy().tolist()
            json_dict['leye_pose'] = np.zeros((1, 3)).reshape(-1).tolist()
            json_dict['reye_pose'] = np.zeros((1, 3)).reshape(-1).tolist()

            tt = -J_0[:,0,:] + torch.cat([opt_offset_x, opt_offset_y, torch.zeros(1, device=device)], dim=0)
            json_dict['transl'] = tt.reshape(-1).detach().cpu().numpy().tolist()
            json_dict['scale'] = opt_scale.detach().cpu().numpy().tolist()


            with open(os.path.join(debug_folder, f'{file_name}.json'), 'w') as f:
                json.dump(json_dict, f, indent=4)


            save_mask = (255 * soft_mask[0]).data.cpu().detach().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(debug_folder, f'{file_name}_mask.png'), save_mask)
            save_gt_mask = (255 * gt_mask[0]).data.cpu().detach().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(debug_folder, f'{file_name}_gt_mask.png'), save_gt_mask)

            V_2d = V[0,:,:2].detach().cpu().numpy() * np.array([[args.size // 2, -args.size // 2]]) + np.array([args.size // 2, args.size // 2])

            overlap_img = vis_meshes(np_img, V_2d, alpha=0.8, radius=2, color=(0, 0, 255))
            overlap_img = cv2.cvtColor(overlap_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(debug_folder, f'{file_name}_fit.png'), overlap_img)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-path", default='./data/examples/images', type=str, help="Input RGBA path")
    parser.add_argument("-o", "--output-path", default='./data/examples/smplx', type=str, help="Output path")

    parser.add_argument("--size", default=1024, type=int, help="Render images size")
    parser.add_argument("--debug", action='store_true', help="Debug mode")

    parser.add_argument("--opt_orient", action='store_true', help="Optimize global orientation")
    parser.add_argument("--opt_pose", action='store_true', help="Optimize body pose")
    parser.add_argument("--opt_betas", action='store_true', help="Optimize shape parameters")

    args = parser.parse_args()

    main(args)

