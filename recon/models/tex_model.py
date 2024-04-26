"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""

import logging as log
import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding
from .networks.encoder import HGFilter
from .networks.mlps import MLP
from .networks.layers import get_activation_class, get_layer_class



class TexModel(nn.Module):

    def __init__(self, config, smpl_query):

        super().__init__()

        self.cfg = config
        self.query = smpl_query
        self._init_model()

    def _init_model(self):
        """Initialize model.
        """
        log.info("Initializing PIFu feature encoder...")

        self.image_encoder = HGFilter(1, 4, 3, self.cfg.feat_dim, 'group', 'no_down', False)

        log.info("Initializing neural field...")

        if self.cfg.color_freq != 0 :
            self.rgb_embedder = PositionalEncoding(self.cfg.color_freq, self.cfg.color_freq -1, input_dim=self.cfg.pos_dim)
            self.rgb_embedder = self.rgb_embedder.to(self.device)
            self.embed_dim  = self.rgb_embedder.out_dim
        else:
            self.rgb_embedder = None
            self.embed_dim = self.cfg.pos_dim

        self.rgb_input_dim = self.embed_dim + self.cfg.feat_dim * 2

        self.rgb_decoder = MLP(self.rgb_input_dim, 3, activation=get_activation_class(self.cfg.activation),
                                    bias=True, layer=get_layer_class(self.cfg.layer_type), num_layers=self.cfg.num_layers-1,
                                    hidden_dim=self.cfg.hidden_dim // 2)


    def _query_feature(self, feat, uv):
        '''
        extract image features at floating coordinates with bilinear interpolation
        args:
            feat: [B, C, H, W] image features
            uv: [B, N, 2] normalized image coordinates ranged in [-1, 1]
        return:
            [B, N, C] sampled pixel values
        '''
        xy = torch.clip(uv, -1, 1).clone()
        xy[..., 1] = -xy[..., 1]

        samples = torch.nn.functional.grid_sample(feat, xy.unsqueeze(2), align_corners=True)
        return samples[:, :, :, 0].permute(0, 2, 1)


    def _get_pos_features(self, pts, smpl_v, vis_class, embedder=None, pos_dim=1):
        with torch.no_grad():
            if pos_dim == 1:
                coord_feats = pts[:, :, 2:3]
            elif pos_dim == 3:
                coord_feats = pts
            elif pos_dim == 5:
                out_coord, sdf, normal, v = self.query.interpolate_vis(pts, smpl_v, vis_class)
                coord_feats = torch.cat([out_coord, sdf, v], dim=-1)
            elif pos_dim == 6:
                out_coord, sdf, normal, v = self.query.interpolate_vis(pts, smpl_v, vis_class)
                coord_feats = torch.cat([out_coord, sdf, v, pts[:, :, 2:3]], dim=-1)
            else:
                out_coord, sdf, normal, v = self.query.interpolate_vis(pts, smpl_v, vis_class)
                coord_feats = torch.cat([out_coord, sdf, normal, v], dim=-1) 

            if embedder is not None:
                coord_feats = embedder(coord_feats)
                
        return coord_feats

    def _get_local_features(self, pts, front_img, back_img):

        front_img_feats = self._query_feature(front_img, pts[:, :, :2])
        back_img_feats = self._query_feature(back_img, pts[:, :, :2])

        return torch.cat([front_img_feats, back_img_feats], dim=-1)


    def compute_rgb(self, x, front_feat, back_feat, smpl_v, vis_class):

        rgb_feats = self._get_local_features(x, front_feat, back_feat)


        coord_feats = self._get_pos_features(x, smpl_v, vis_class, self.rgb_embedder, self.cfg.pos_dim)

        return self.rgb_decoder(torch.cat([rgb_feats, coord_feats], dim=-1), sigmoid=True)


    def compute_feat_map(self, front_img, back_img):
                
        front_feat = self.image_encoder(front_img)
        back_feat = self.image_encoder(back_img)
        return front_feat, back_feat


    def forward(self, front_feat, back_feat, pts, smpl_v, vis_class):

        pred_rgb = self.compute_rgb(pts, front_feat, back_feat, smpl_v, vis_class)

        return pred_rgb