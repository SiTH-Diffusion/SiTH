"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import io
import math
import h5py
import PIL.Image as Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

class TrainDiffDataset(Dataset):
    def __init__(self, args):
        self.img_size = args.resolution

        self.back_only = not args.sample_random_views
        self.aug_bg = not args.white_background

        self._init_from_h5(args.train_data_dir)

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
            except:
                raise ValueError("[Error] Can't load from h5 dataset")
        self.initialization_mode = "h5"

    def _augment_background(self, image, mask, color, clip=False):
        # Random background
        if clip:
            bg_color = (color - CLIP_MEAN) / CLIP_STD
        else:
            bg_color = (color - 0.5) / 0.5

        bg = torch.ones_like(image) * bg_color.view(3,1,1)
        _mask = ~(mask.bool()).expand_as(image)
        image[_mask] = bg[_mask]

        return image

    def __getitem__(self, idx: int):
        """Retrieve point sample."""
        if self.initialization_mode is None:
            raise Exception("The dataset is not initialized.")
        
        view_ids = torch.randperm(self.num_views)[:2]
        if self.back_only:
            view_ids[1] = ( view_ids[0] + self.num_views // 2 ) % self.num_views

        return self._get_h5_data(idx, view_ids)
    
    def _get_h5_data(self, subject_id, view_ids):
        with h5py.File(self.h5_path, "r") as f:
            try:    
                sub = f[self.subject_names[subject_id]]

                src_pil = Image.open(io.BytesIO(sub['rgb_img'][view_ids[0]]))
                tar_pil = Image.open(io.BytesIO(sub['rgb_img'][view_ids[1]]))

                bg_color = torch.ones((3, 1))
                if self.aug_bg:
                    bg_color = torch.rand(3)

                # Load images
                src_clip_rgba = self.transform_clip(src_pil)
                src_clip_mask = src_clip_rgba[-1:,...]
                src_clip_rgb = src_clip_rgba[:-1,...]
                src_clip_image = self._augment_background(src_clip_rgb, src_clip_mask, bg_color, clip=True)

                src_rgba = self.transform_rgba(src_pil)
                src_mask = src_rgba[-1:,...]
                src_rgb = src_rgba[:-1,...]
                src_image = self._augment_background(src_rgb, src_mask, bg_color)

                tgt_rgba = self.transform_rgba(tar_pil)
                tgt_mask = tgt_rgba[-1:,...]
                tgt_rgb = tgt_rgba[:-1,...]
                tgt_image = self._augment_background(tgt_rgb, tgt_mask, bg_color)

                # Load uv maps
                tgt_uv = Image.open(io.BytesIO(sub['uv_img'][view_ids[1]]))


                # Load camera pos

                diff_elevation_deg = torch.tensor(sub['cam_eva'][view_ids[1]] - sub['cam_eva'][view_ids[0]])
                diff_azimuth_deg = torch.tensor(sub['cam_azh'][view_ids[1]] - sub['cam_azh'][view_ids[0]])
                diff_radius = torch.tensor(sub['cam_rad'][view_ids[1]] - sub['cam_rad'][view_ids[0]])

                diff_elevation = diff_elevation_deg * math.pi / 180
                diff_azimuth = diff_azimuth_deg * math.pi / 180

                view_cond = torch.stack(
                    [   diff_elevation,
                        torch.sin(diff_azimuth),
                        torch.cos(diff_azimuth),
                        diff_radius] ).view(-1,1,1).repeat(1, self.img_size, self.img_size)

            except:
                raise ValueError("[Error] Can't read key (%s, %s) from h5 dataset" % (subject_id, view_ids))
        
        return {
            'src_ori_image': src_image,
            'src_image': src_clip_image,
            'tgt_image': tgt_image,
            'tgt_mask': tgt_mask,
            'tgt_uv':  self.transform(tgt_uv),
            'view_cond': view_cond      
        }

    def __len__(self):
        return self.num_subjects
