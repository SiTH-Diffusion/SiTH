"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import io 
import math
import PIL.Image as Image
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class TrainReconDataset(Dataset):
    def __init__(self, root, cfg):
        self.root = root
        self.img_size = cfg.img_size
        self.num_samples = cfg.num_samples
        self.aug_jitter = cfg.aug_jitter
        self.aug_bg = not cfg.white_bg

        self._init_from_h5(self.root)

        self.transform_rgba= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5,0.0], std=[0.5,0.5,0.5,1.0])
        ])
        self.transform= transforms.Compose([
            transforms.Resize((self.img_size, self.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
        ])
        self.jitter = transforms.ColorJitter(brightness=.5, contrast=.5, saturation=.5, hue=.25)


    def _init_from_h5(self, dataset_path):
        """Initializes the dataset from a h5 file.
           copy smpl_v from h5 file.
        """
        self.h5_path = dataset_path
        with h5py.File(dataset_path, "r") as f:
            try:
                self.num_subjects = len(list(f))
                self.subject_names = [x for x in list(f)]
                sub = f[self.subject_names[0]]  
                self.num_views = sub['cam_eva'].shape[0]
                self.num_pts = sub['d'].shape[0]
            except:
                raise ValueError("[Error] Can't load from h5 dataset")
        self.initialization_mode = "h5"

    def _rotation_matrix(self, azimuth_deg, elevation_deg):
        # Convert degrees to radians
        theta = math.radians(azimuth_deg)
        phi = math.radians(elevation_deg)
    
        # Azimuth: Rotation about y-axis
        Ry = torch.tensor([
            [math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta), 0, math.cos(theta)]
        ])
    
        # Elevation: Rotation about x-axis
        Rx = torch.tensor([
            [1, 0, 0],
            [0, math.cos(phi), -math.sin(phi)],
            [0, math.sin(phi), math.cos(phi)]
        ])
    
        # Combined rotation matrix
        R = torch.mm(Ry, Rx)
        return R


    def _add_background(self, image, mask, bg_color):
        # Random background
        bg = torch.ones_like(image) * bg_color.view(3,1,1)
        _mask = (mask<0.5).expand_as(image)
        image[_mask] = bg[_mask]
        return image

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")
        
        # points id need to be in accending order
        pts_id = np.random.randint(self.num_pts - self.num_samples, size=1)

        view_id = int(np.random.randint(self.num_views, size=1))

        return self._get_h5_data(idx, np.arange(pts_id, pts_id + self.num_samples), view_id)


    def _get_h5_data(self, subject_id, pts_id, view_id):
        with h5py.File(self.h5_path, "r") as f:
            try:
                back_view_id = (view_id + self.num_views // 2 ) % self.num_views


                sub = f[self.subject_names[subject_id]]
                pts = torch.from_numpy(np.array(sub['pts'][pts_id]))
                d = torch.from_numpy(np.array(sub['d'][pts_id]))
                nrm = torch.from_numpy(np.array(sub['nrm'][pts_id]))
                rgb = torch.from_numpy( np.array(sub['rgb'][pts_id]))

                smpl_v = torch.from_numpy(np.array(sub['smpl_v']))
                front_vis = torch.from_numpy(np.array(sub['vis'][view_id]).astype(bool))
                back_vis = torch.from_numpy(np.array(sub['vis'][back_view_id]).astype(bool))

                vis_class = torch.zeros_like(front_vis, dtype=torch.float32)
                # assign vis class to 1 if front_vis, -1 if back_vis, 0 if none
                vis_class[front_vis] = 1.0
                vis_class[back_vis] = -1.0

                azh = np.array(sub['cam_azh'][view_id])
                eva = np.array(sub['cam_eva'][view_id])

                R = self._rotation_matrix(-azh, -eva)

                # Transform points
                pts = pts @ R.t()
                nrm = nrm @ R.t()
                smpl_v = smpl_v @ R.t() 

                bg_color = torch.ones((3, 1))
                if self.aug_bg:
                    bg_color = (torch.rand(3).float() - 0.5) / 0.5 


                front_pil = Image.open(io.BytesIO(sub['rgb_img'][view_id]))
                front_rgba = self.transform_rgba(front_pil)
                front_mask = front_rgba[-1:,...]
                front_rgb = front_rgba[:-1,...]
                front_image = self._add_background(front_rgb, front_mask, bg_color)


                back_pil = Image.open(io.BytesIO(sub['rgb_img'][back_view_id]))
                back_rgba = torch.flip( self.transform_rgba(back_pil), [2])
                back_mask = back_rgba[-1:,...]
                back_rgb = back_rgba[:-1,...]
                back_image = self._add_background(back_rgb, back_mask, bg_color)


                input_size = front_image.shape[1:]
    
                front_nrm_pil = Image.open(io.BytesIO(sub['nrm_img'][view_id]))
                front_nrm_img = self.transform_rgba(front_nrm_pil) [:3,...]
                front_nrm_img = (R @ front_nrm_img.view(3, -1)).view(3, input_size[0], input_size[1]) * front_mask

                back_nrm_pil = Image.open(io.BytesIO(sub['nrm_img'][back_view_id]))
                back_nrm_img = torch.flip(self.transform_rgba(back_nrm_pil), [2]) [:3,...]
                back_nrm_img = (R @ back_nrm_img.view(3, -1)).view(3, input_size[0], input_size[1]) * back_mask

                if self.aug_jitter:
                    if torch.rand(1) > 0.5:
                        front_image = self.jitter(front_image)
                        back_image = self.jitter(back_image)
            except:
                raise ValueError("[Error] Can't read key (%s, %s, %s) from h5 dataset" % (subject_id, pts_id, view_id))

        return {
                'pts' : pts, 'd' : d, 'nrm' : nrm, 'rgb' : rgb, 'idx' : subject_id,
                'smpl_v' : smpl_v, 'R': R,
                'vis_class': vis_class.unsqueeze(-1),
                'front_rgb_img' : front_image,
                'back_rgb_img' : back_image,
                'front_mask': front_mask,
                'back_mask': back_mask,
                'front_nrm_img' : front_nrm_img,
                'back_nrm_img' : back_nrm_img,
        }
    
    def __len__(self):
        """Return length of dataset (number of _samples_)."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")

        return self.num_subjects
