"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""
import torch
import torch.nn as nn
import time
import trimesh
import os
import logging as log
import cv2
import numpy as np
import nvdiffrast.torch as dr
from PIL import Image
from kaolin.ops.conversions import voxelgrids_to_trianglemeshes
from kaolin.ops.mesh import subdivide_trianglemesh

from .networks.normal_predictor import define_G
from .geo_model import GeoModel
from .tex_model import TexModel
from .SMPL_query import SMPL_query

class Evaluator(nn.Module):

    def __init__(self, config, watertight, can_V, trainer=None):

        super().__init__()

        # Set device to use
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.glctx = dr.RasterizeCudaContext(device=self.device)

        #device_name = torch.cuda.get_device_name(device=self.device)
        #log.info(f'Using {device_name} with CUDA v{torch.version.cuda}')

        self.cfg = config

        # create marching cube grid
        self.res = self.cfg.grid_size
        window_x = torch.linspace(-1., 1., steps=self.res, device='cuda')
        window_y = torch.linspace(-1., 1., steps=self.res, device='cuda')
        window_z = torch.linspace(-1., 1., steps=self.res, device='cuda')

        self.coord = torch.stack(torch.meshgrid(window_x, window_y, window_z, indexing='ij')).permute(
                                1, 2, 3, 0).reshape(1, -1, 3).contiguous()

        self.smpl_query = SMPL_query(watertight['smpl_F'], can_V)

        if trainer is None:
            self.nrm_predictor = define_G(3, 3, 64, "global", 4, 9, 1, 3, "instance")

            self.geo_model = GeoModel(self.cfg, self.smpl_query)
            self.tex_model = TexModel(self.cfg, self.smpl_query)

            checkpoint = torch.load(self.cfg.resume, map_location=self.device)

            self.nrm_predictor.load_state_dict(checkpoint['nrm_encoder'])
            self.geo_model.image_encoder.load_state_dict(checkpoint['image_encoder'])
            self.geo_model.sdf_decoder.load_state_dict(checkpoint['sdf_decoder'])

            self.tex_model.image_encoder.load_state_dict(checkpoint['high_res_encoder'])
            self.tex_model.rgb_decoder.load_state_dict(checkpoint['rgb_decoder'])

            self.nrm_predictor.to(self.device)
            self.geo_model.to(self.device)
            self.tex_model.to(self.device)
        else:
            self.nrm_predictor = trainer.nrm_predictor
            self.geo_model = trainer.geo_model
            self.tex_model = trainer.tex_model

    def _repair_mesh(self, mesh):

        # remove disconnect par of mesh
        connected_comp = mesh.split(only_watertight=False)
        max_area = 0
        max_comp = None
        for comp in connected_comp:
            if comp.area > max_area:
                max_area = comp.area
                max_comp = comp
        mesh = max_comp
            
        trimesh.repair.fix_inversion(mesh)

        return mesh

    def _uv_padding(self, image, hole_mask):
        inpaint_image = (
            cv2.inpaint(
                (image.detach().cpu().numpy() * 255).astype(np.uint8),
                (hole_mask.detach().cpu().numpy() * 255).astype(np.uint8),
                4,
                cv2.INPAINT_TELEA,
                )
            )
        return inpaint_image

    def test_reconstruction(self, data, save_path, subdivide=True, chunk_size=1e5, flip=False, save_uv=False):

        fname = data['fname'][0]
        log.info(f"Reconstructing mesh for {fname}...")

        start = time.time()

        front_rgb_img = data['front_rgb_img'].cuda()
        back_rgb_img = data['back_rgb_img'].cuda()
        smpl_v = data['smpl_v'].cuda()
        vis_class = data['vis_class'].cuda()


        b, c, h, w = front_rgb_img.shape

        front_nrm_img = self.nrm_predictor(front_rgb_img)
        back_nrm_img = self.nrm_predictor(back_rgb_img)
        
        front_feat, back_feat = self.geo_model.compute_feat_map(front_rgb_img, back_rgb_img, front_nrm_img, back_nrm_img)


        # first estimate the sdf values
        _points = torch.split(self.coord, int(chunk_size), dim=1)
        voxels = []
        for _p in _points:
            pred_sdf = self.geo_model.compute_sdf(_p, front_feat, back_feat, smpl_v, vis_class)
            voxels.append(pred_sdf)

        voxels = torch.cat(voxels, dim=1)[..., 0]
        voxels = voxels.reshape(1, self.res, self.res, self.res)
        
        vertices, faces = voxelgrids_to_trianglemeshes(voxels, iso_value=0.)
        vertices = ((vertices[0].reshape(1, -1, 3) - 0.5) / (self.res/2)) - 1.0
        faces = faces[0]

        if subdivide:
            vertices, faces = subdivide_trianglemesh(vertices, faces, iterations=1)

        # Next estimate the texture rgb values on the surface
        if save_uv:
            
            d = trimesh.Trimesh(vertices=vertices[0].cpu().detach().numpy(), 
                faces=faces.cpu().detach().numpy(), 
                process=False)
            
            d = self._repair_mesh(d)

            import xatlas

            vmap, uv_faces, uvs = xatlas.parametrize(d.vertices, d.faces)
            faces_tensor = torch.from_numpy(uv_faces.astype(np.int32)).to(self.device)

            uv_clip = torch.from_numpy(uvs) * torch.tensor([2.0, -2.0]) + torch.tensor([-1.0, 1.0])
            # pad to four component coordinate
            uv_clip4 = torch.cat(
                (
                    uv_clip,
                    torch.zeros_like(uv_clip[..., 0:1]),
                    torch.ones_like(uv_clip[..., 0:1]),
                ),
                dim=-1,
            ).to(self.device)
            # rasterize
            rast, _ = dr.rasterize(self.glctx, uv_clip4[None,...], faces_tensor, (1024, 1024), grad_db=True)
            #rast0 = torch.flip(rast[0], dims=(1,))
            rast0 = rast[0]
            hole_mask = ~(rast0[:, :, 3] > 0)
            gb_pos, _ = dr.interpolate(torch.from_numpy(d.vertices).to(self.device).float(), rast, torch.from_numpy(d.faces).to(self.device).int())
            gb_pos = gb_pos[0].view(1,-1, 3)
            _points = torch.split(gb_pos, int(chunk_size), dim=1)

            front_feat, back_feat = self.tex_model.compute_feat_map(front_rgb_img, back_rgb_img)
            pred_rgb = []
            for _p in _points:
                output = self.tex_model.compute_rgb(_p, front_feat, back_feat, smpl_v, vis_class)
                pred_rgb.append(output)

            pred_rgb = torch.cat(pred_rgb, dim=1).view(1024, 1024, 3)

            pad_rgb = self._uv_padding(pred_rgb, hole_mask)

            texture_map = Image.fromarray(pad_rgb)

            h = trimesh.Trimesh(
                vertices=d.vertices[vmap],
                faces=uv_faces,
                visual=trimesh.visual.TextureVisuals(uv=uvs, image=texture_map),
                process=False,
            )

            if flip: # flip to the gradio coordinate system
                h.apply_transform( [[-1, 0, 0, 0],
                                [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]] )

            h.visual.material.name = fname
            obj_path = os.path.join(save_path, '%s_reco.obj' % (fname))
            h.export(obj_path, mtl_name=fname+'.mtl')

            with open(os.path.join(save_path, fname+'.mtl'), 'w') as f:
                f.write('newmtl {}\n'.format(fname))
                f.write('map_Kd {}.png\n'.format(fname))

        else:
            _points = torch.split(vertices, int(chunk_size), dim=1)
            front_feat, back_feat = self.tex_model.compute_feat_map(front_rgb_img, back_rgb_img)
            pred_rgb = []
            for _p in _points:
                output = self.tex_model.compute_rgb(_p, front_feat, back_feat, smpl_v, vis_class)
                pred_rgb.append(output)

            pred_rgb = torch.cat(pred_rgb, dim=1)
        
            h = trimesh.Trimesh(vertices=vertices[0].cpu().detach().numpy(), 
                faces=faces.cpu().detach().numpy(), 
                vertex_colors=pred_rgb[0].cpu().detach().numpy(),
                process=False)

            if flip: # flip to the gradio coordinate system
                h.apply_transform( [[-1, 0, 0, 0],
                                [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]] )


            h = self._repair_mesh(h)
            obj_path = os.path.join(save_path, '%s_reco.obj' % (fname))
            h.export(obj_path)

        end = time.time()
        log.info(f"Reconstruction finished in {end-start} seconds.")
        
        return obj_path


