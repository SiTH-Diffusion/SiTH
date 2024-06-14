"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""

import torch
import torch.nn as nn
from .ops.mesh import *


class SMPL_query(nn.Module):

    def __init__(self, smpl_F, can_V):
        super().__init__()
        self.smpl_F = smpl_F.cuda() #[num_faces, 3]
        self.uv = can_V.unsqueeze(0).cuda() #[1, num_vertices, 3]

    def interpolate(self, coords, smpl_V):

        """Query local features using the feature codebook, or the given input_code.
        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            smpl_V (torch.FloatTensor): SMPL vertices of shape [batch, num_vertices, 3]
        Returns:
            (torch.FloatTensor): interpolated features of shape [batch, num_samples, feature_dim]
        """
        b_size = coords.shape[0]
        sdf, hitpt, fid, weights = batched_closest_point_fast(smpl_V, self.smpl_F,
                                                              coords) # [B, Ns, 1], [B, Ns, 3], [B, Ns, 1], [B, Ns, 3]
        
        normal = torch.nn.functional.normalize( hitpt - coords, eps=1e-6, dim=2) # [B x Ns x 3]
        hitface = self.smpl_F[fid] # [B, Ns, 3]

        inputs_feat = self.uv.repeat(b_size, 1, 1).unsqueeze(2).expand(-1, -1, hitface.shape[-1], -1) 
            
        indices = hitface.unsqueeze(-1).expand(-1, -1, -1, inputs_feat.shape[-1])
        nearest_feats = torch.gather(input=inputs_feat, index=indices, dim=1) # [B, Ns, 3, 3]

        out_coord = torch.sum(nearest_feats * weights[...,None], dim=2) # K-weighted sum by: [B x Ns x 3]
        
        #coords_feats = torch.cat([out_coord, sdf, normal, coords[...,2:3]], dim=-1) # [B, Ns, 8]
        z = coords[...,2:3]
        return out_coord, sdf, normal, z
    
    def interpolate_vis(self, coords, smpl_V, vis):

        """Query local features using the feature codebook, or the given input_code.
        Args:
            coords (torch.FloatTensor): coords of shape [batch, num_samples, 3]
            smpl_V (torch.FloatTensor): SMPL vertices of shape [batch, num_vertices, 3]
            vis (torch.FloatTensor): visibility of shape [batch, num_faces, 1]
        Returns:
            (torch.FloatTensor): interpolated features of shape [batch, num_samples, feature_dim]
        """
        b_size = coords.shape[0]
        sdf, hitpt, fid, weights = batched_closest_point_fast(smpl_V, self.smpl_F,
                                                              coords) # [B, Ns, 1], [B, Ns, 3], [B, Ns, 1], [B, Ns, 3]
        
        normal = torch.nn.functional.normalize( hitpt - coords, eps=1e-6, dim=2) # [B x Ns x 3]
        hitface = self.smpl_F[fid] # [B, Ns, 3]

        vismap = torch.gather(input=vis, index=fid.unsqueeze(-1), dim=1) # [B, Ns, 1]

        inputs_feat = self.uv.repeat(b_size, 1, 1).unsqueeze(2).expand(-1, -1, hitface.shape[-1], -1) # B, Ns, 3, 3
            
        indices = hitface.unsqueeze(-1).expand(-1, -1, -1, inputs_feat.shape[-1]) # B, Ns, 3, 3
        nearest_feats = torch.gather(input=inputs_feat, index=indices, dim=1) # [B, Ns, 3, 3]

        out_coord = torch.sum(nearest_feats * weights[...,None], dim=2) # K-weighted sum by: [B x Ns x 3]
        
        #coords_feats = torch.cat([out_coord, sdf, normal, coords[...,2:3]], dim=-1) # [B, Ns, 8]
        #z = coords[...,2:3]
        return out_coord, sdf, normal, vismap

