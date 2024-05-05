"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import json
import torch
from smplx import SMPLX

JOINT_MAP = 'data/smplx_openpose25.json'
SMPL_PATH = 'data/body_models/smplx'


def load_json(json_path, dataset_name='thuman', device='cuda'):
    smpl_data = json.load(open(json_path))
    mapper = json.load(open(JOINT_MAP))['smplx_idxs']

    param_betas = torch.tensor(smpl_data['betas'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_poses = torch.tensor(smpl_data['body_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_left_hand_pose = torch.tensor(smpl_data['left_hand_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_right_hand_pose = torch.tensor(smpl_data['right_hand_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
            
    param_expression = torch.tensor(smpl_data['expression'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_jaw_pose = torch.tensor(smpl_data['jaw_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_leye_pose = torch.tensor(smpl_data['leye_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_reye_pose = torch.tensor(smpl_data['reye_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    

    if dataset_name=='customhumans':
        flat_hand_mean = True
        gender = 'male'
    elif dataset_name=='thuman':
        flat_hand_mean = False
        gender = 'male'
    else:
        flat_hand_mean = False
        gender = smpl_data.get('gender', 'neutral')


    body_model = SMPLX(model_path=SMPL_PATH, gender=gender, use_pca=True, num_pca_comps=12,
                        flat_hand_mean=flat_hand_mean, use_face_contour=True).to(device)

    
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

    V = output.vertices.detach().to(device)
    F = torch.tensor(body_model.faces.astype(int)).to(device)
    joint_3d = output.joints.detach().to(device)[:, mapper]

    return V, F, joint_3d