"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import argparse
import torch
import numpy as np
import os
import trimesh
from PIL import Image
import pickle
import cv2
from tqdm import tqdm
from smplx import SMPLX
import json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SMPLX_PATH = 'data/body_models/smplx'

body_model = SMPLX(model_path=SMPLX_PATH, gender='male', use_pca=True, num_pca_comps=12,
                        flat_hand_mean=False, use_face_contour=True).to(device)


def main(args):

    for id in tqdm(range(526)):
        name_id = "%04d" % id
        # TODO: Note that you need to rename the THuman2.0 mesh and smplx folders
        input_file = os.path.join(args.input_path, 'THuman2.0', name_id, name_id + '.obj')
        tex_file = os.path.join(args.input_path, 'THuman2.0', name_id, 'material0.jpeg')
        smpl_file = os.path.join(args.input_path, 'THuman2.0_smplx', name_id, 'smplx_param.pkl')

        smpl_data = pickle.load(open(smpl_file,'rb'))
        out_file_name = os.path.splitext(os.path.basename(input_file))[0]
        output_aligned_path = os.path.join(args.output_path, out_file_name)
        os.makedirs(output_aligned_path, exist_ok=True)


        textured_mesh = trimesh.load(input_file)

        output = body_model(body_pose = torch.tensor(smpl_data['body_pose'], device=device),
                                    betas = torch.tensor(smpl_data['betas'], device=device),
                                    left_hand_pose = torch.tensor(smpl_data['left_hand_pose'], device=device),
                                    right_hand_pose = torch.tensor(smpl_data['right_hand_pose'], device=device),
                                    expression = torch.tensor(smpl_data['expression'], device=device),
                                    jaw_pose = torch.tensor(smpl_data['jaw_pose'], device=device),
                                    leye_pose = torch.tensor(smpl_data['leye_pose'], device=device),
                                    reye_pose = torch.tensor(smpl_data['reye_pose'], device=device)
                                   )
        
        J_0 = output.joints.detach().cpu().numpy()[0,0,:]
        d = trimesh.Trimesh(vertices=output.vertices.detach().cpu().numpy()[0] -J_0 ,
                                        faces=body_model.faces)

        R = np.asarray(smpl_data['global_orient'][0])
        rot_mat = np.zeros(shape=(3,3))
        rot_mat, _ = cv2.Rodrigues(R)
        scale = smpl_data['scale']

        T = -np.asarray(smpl_data['translation'])
        S = np.eye(4)
        S[:3, 3] = T
        textured_mesh.apply_transform(S)

        S = np.eye(4)
        S[:3, :3] *= 1./scale
        textured_mesh.apply_transform(S)

        T = -J_0
        S = np.eye(4)
        S[:3, 3] = T
        textured_mesh.apply_transform(S)

        S = np.eye(4)
        S[:3, :3] = np.linalg.inv(rot_mat)
        textured_mesh.apply_transform(S)

        visual = trimesh.visual.texture.TextureVisuals(uv=textured_mesh.visual.uv, image=Image.open(tex_file))

        t = trimesh.Trimesh(vertices=textured_mesh.vertices,
                                     faces=textured_mesh.faces,
                                     vertex_normals=textured_mesh.vertex_normals,
                                     visual=visual,
                                     process=False)
                            
        d.export(os.path.join(output_aligned_path, out_file_name + '_smplx.obj')  )
        t.export(os.path.join(output_aligned_path, out_file_name + '.obj')  )
        with open(os.path.join(output_aligned_path, 'material.mtl'), 'w') as f:
                f.write('newmtl material_0\n'.format(out_file_name))
                f.write('map_Kd material_0.jpeg\n'.format(out_file_name))

        result = {}
        result ['transl'] = (-J_0).tolist()
        for key, val in smpl_data.items():
            if key not in ['scale', 'translation']:
                result[key] = val[0].tolist()

        json_file = os.path.join(output_aligned_path, out_file_name + '_smplx.json')
        json.dump(result, open(json_file, 'w'), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Align THuman2.0 dataset with SMPLX')
    parser.add_argument('-i', "--input-path", default='data/THuman', type=str, help="Input path")
    parser.add_argument('-o', "--output-path", default='data/THuman/new_thuman', type=str, help="Output path")

    args = parser.parse_args()
    main(args)
    