"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""

import os 
import math

import PIL.Image as Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import kaolin as kal
import nvdiffrast

from .load_obj import load_obj

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
glctx = nvdiffrast.torch.RasterizeCudaContext(device=device)


class TestDiffDataset(Dataset):
    def __init__(self, root, uv_template, size=512):
        self.root = root
        self.img_size = size

        _, _, texv, texf, _ = load_obj(uv_template, load_materials=True)

        self.texv = texv.to(device)
        self.texf = texf.to(device)

        self.front_folder = os.path.join(self.root, 'images')
        self.smplx_folder = os.path.join(self.root, 'smplx')

        # RGBA
        self.subject_list = [os.path.join(self.front_folder, x) for x in sorted(os.listdir(self.front_folder)) if x.endswith(('.png', '.jpg'))]
        # smplx obj
        self.smplx_list = [os.path.join(self.smplx_folder, x) for x in sorted(os.listdir(self.smplx_folder)) if x.endswith('.obj')]


        assert len(self.subject_list) == len(self.smplx_list)

        self.num_subjects = len(self.subject_list)


        #  set camera
        look_at = torch.zeros( (1, 3), dtype=torch.float32, device=device)
        camera_up_direction = torch.tensor( [[0, 1, 0]], dtype=torch.float32, device=device)
        camera_position = torch.tensor( [[0, 0, -3]], dtype=torch.float32, device=device)


        self.camera = kal.render.camera.Camera.from_args(eye=camera_position,
                                         at=look_at,
                                         up=camera_up_direction,
                                         width=self.img_size, height=self.img_size,
                                         near=-512, far=512,
                                        fov_distance=1.0, device=device)


        self.transform= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])

        self.transform_rgba= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5,0.0], std=[0.5,0.5,0.5,1.0])
        ])
        self.transform_clip = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073, 0.0],
                                 [0.26862954, 0.26130258, 0.27577711, 1.0]),
        ])


    def render_uv_map(self, camera, V, F, texv, texf, size):

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

    def __getitem__(self, index):


        # Load images
        src_rgba = self.transform_clip(Image.open(self.subject_list[index]))
        src_mask = src_rgba[-1:,...]
        src_rgb = src_rgba[:-1,...]
        src_clip_image = src_rgb * src_mask


        src_rgba = self.transform_rgba(Image.open(self.subject_list[index]))
        src_mask = src_rgba[-1:,...]
        src_rgb = src_rgba[:-1,...]
        src_image = src_rgb * src_mask


        # load smpl verts
        V, F = load_obj(self.smplx_list[index])
        uv = self.render_uv_map(self.camera, V.to(device), F.to(device), self.texv, self.texf, self.img_size)

        tgt_uv = torch.zeros((self.img_size, self.img_size, 3))

        tgt_uv[..., :2] = uv[0].cpu()

        # view condition is always the same for back images
        view_cond = torch.stack(
            [   torch.tensor(0.0),
                torch.sin(torch.tensor(math.pi)),
                torch.cos(torch.tensor(math.pi)),
                torch.tensor(0.0)] ).view(-1,1,1).repeat(1, self.img_size, self.img_size)

        return {'src_ori_image': src_image,
                'src_image': src_clip_image,
                'tgt_uv':  tgt_uv.permute(2,0,1) * 2. - 1,
                'tgt_mask': torch.flip(src_mask, [2]),
                'view_cond': view_cond,
                'filename': self.subject_list[index].split('/')[-1].split('.')[0]
        }

    def __len__(self):
        return self.num_subjects
