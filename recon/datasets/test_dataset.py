"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""

import os 
import PIL.Image as Image
import numpy as np
import cv2
import nvdiffrast
import kaolin as kal
import torch

from torch.utils.data import Dataset
from torchvision import transforms

from .load_obj import load_obj

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)

class TestFolderDataset(Dataset):
    def __init__(self, root, cfg, watertight):
        self.root = root

        self.front_folder = os.path.join(self.root, 'images')
        self.back_folder = os.path.join(self.root, 'back_images')
        self.smplx_folder = os.path.join(self.root, 'smplx')

        # RGBA
        self.subject_list = [os.path.join(self.front_folder, x) for x in sorted(os.listdir(self.front_folder)) if x.endswith(('.png', '.jpg'))]
        # smplx obj
        self.smplx_list = [os.path.join(self.smplx_folder, x) for x in sorted(os.listdir(self.smplx_folder)) if x.endswith('.obj')]
        # RGB
        self.back_list = [os.path.join(self.back_folder, x) for x in sorted(os.listdir(self.back_folder)) if x.endswith(('.png', '.jpg'))]

        assert len(self.subject_list) == len(self.smplx_list) == len(self.back_list)

        self.num_subjects = len(self.subject_list)

        self.img_size = cfg.img_size
        self.erode_iter = cfg.erode_iter

        self.F = watertight['smpl_F'].to(device)

        #  set camera
        look_at = torch.zeros( (2, 3), dtype=torch.float32, device=device)
        camera_up_direction = torch.tensor( [[0, 1, 0]], dtype=torch.float32, device=device).repeat(2, 1,)
        camera_position = torch.tensor( [[0, 0, 3],
                                        [0, 0, -3]], dtype=torch.float32, device=device)


        self.camera = kal.render.camera.Camera.from_args(eye=camera_position,
                                         at=look_at,
                                         up=camera_up_direction,
                                         width=self.img_size, height=self.img_size,
                                         near=-512, far=512,
                                        fov_distance=1.0, device=device)


        self.transform= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.to_tensor= transforms.Compose([
            transforms.ToTensor(),
        ])    

    def render_visiblity(self, camera, V, F, size=(1024, 1024)):

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



    def add_background(self, image, mask, color=[0.0, 0.0, 0.0]):
        # Random background
        bg_color = torch.tensor(color).float()
        bg = torch.ones_like(image) * bg_color.view(3,1,1)
        _mask = (mask<0.5).expand_as(image)
        image[_mask] = bg[_mask]
        return image
    

    def erode_mask(self, mask, kernal=(5,5), iter=1):
        mask = torch.from_numpy(cv2.erode(mask[0].numpy(), np.ones(kernal, np.uint8), iterations=iter)).float().unsqueeze(0)
        return mask


    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        

        # load rgba image
        front_img = Image.open(self.subject_list[idx])
        front_img = self.transform(front_img)
        front_rgb = front_img[:3,...]
        mask = front_img[3:,...]

        # erode mask, this is to remove noise near the boundary
        if self.erode_iter > 0:
            mask =  self.erode_mask(mask, kernal=(5,5), iter=self.erode_iter)
        
        front_rgb = self.add_background(front_rgb, mask)


        back_img = Image.open(self.back_list[idx]).convert('RGB')
        back_img = torch.flip(self.transform(back_img), [2])
        back_img = self.add_background(back_img, mask)

        
        # load smpl verts
        V, _ = load_obj(self.smplx_list[idx])
        vis = self.render_visiblity(self.camera, V.to(device), self.F)
        
        return {
            'fname': self.subject_list[idx].split('/')[-1].split('.')[0],
            'smpl_v' : V.cpu(),
            'vis_class': vis.cpu(),
            'front_rgb_img': (front_rgb - 0.5) / 0.5,
            'back_rgb_img': (back_img - 0.5) / 0.5,
            'mask': mask,
        }

    def __len__(self):
        """Return length of dataset (number of _samples_)."""

        return self.num_subjects
