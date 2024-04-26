# Adapted from https://github.com/caizhongang/SMPLer-X/blob/main/main/SMPLer_X.py
"""
Copyright 2022 S-Lab
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

from .vit import ViT
from .smpler_x import PositionNet, HandRotationNet, FaceRegressor, BoxNet, HandRoI, BodyRotationNet

from ..utils.transforms import rot6d_to_axis_angle

class Model(nn.Module):
    def __init__(self, ckpt=None):
        super(Model, self).__init__()


        self.input_img_shape = (512, 384)
        self.input_body_shape = (256, 192)
        self.input_hand_shape = (256, 256)
        self.output_hm_shape  = (16, 16, 12)
        self.output_hand_hm_shape= (16, 16, 16)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._init_model(ckpt)

    def _init_model(self, ckpt=None):

        self.encoder = ViT( img_size = (256, 192), patch_size = 16, embed_dim = 1280,
               depth = 32, num_heads = 16, ratio = 1, use_checkpoint = False,
               mlp_ratio = 4, qkv_bias = True, drop_path_rate = 0.55)
        
        self.body_position_net = PositionNet('body', feat_dim=1280)
        self.body_regressor = BodyRotationNet(feat_dim=1280)
        self.box_net = BoxNet(feat_dim=1280)
        self.hand_position_net = PositionNet('hand', feat_dim=1280)
        self.hand_roi_net = HandRoI(feat_dim=1280, upscale=4)
        self.hand_regressor = HandRotationNet('hand', feat_dim=1280)
        self.face_regressor = FaceRegressor(feat_dim=1280)

        if ckpt is not None:
            checkpoint = torch.load(ckpt, map_location='cpu')
            self.encoder.load_state_dict(checkpoint['encoder'])
            self.body_position_net.load_state_dict(checkpoint['body_position_net'])
            self.body_regressor.load_state_dict(checkpoint['body_regressor'])
            self.box_net.load_state_dict(checkpoint['box_net'])
            self.hand_position_net.load_state_dict(checkpoint['hand_position_net'])
            self.hand_roi_net.load_state_dict(checkpoint['hand_roi_net'])
            self.hand_regressor.load_state_dict(checkpoint['hand_regressor'])
            self.face_regressor.load_state_dict(checkpoint['face_regressor'])
            print('Loaded weights from {}'.format(ckpt))

        self.encoder.to(self.device)
        self.body_position_net.to(self.device)
        self.body_regressor.to(self.device)
        self.box_net.to(self.device)
        self.hand_position_net.to(self.device)
        self.hand_roi_net.to(self.device)
        self.hand_regressor.to(self.device)
        self.face_regressor.to(self.device)

    def forward(self, inputs):

        body_img = F.interpolate(inputs['img'], self.input_body_shape)

        # 1. Encoder
        img_feat, task_tokens = self.encoder(body_img)  # task_token:[bs, N, c]
        shape_token, cam_token, expr_token, jaw_pose_token, hand_token, body_pose_token = \
            task_tokens[:, 0], task_tokens[:, 1], task_tokens[:, 2], task_tokens[:, 3], task_tokens[:, 4:6], task_tokens[:, 6:]

        # 2. Body Regressor
        body_joint_hm, body_joint_img = self.body_position_net(img_feat)
        root_pose, body_pose, shape, cam_param, = self.body_regressor(body_pose_token, shape_token, cam_token, body_joint_img.detach())
        root_pose = rot6d_to_axis_angle(root_pose)
        body_pose = rot6d_to_axis_angle(body_pose.reshape(-1, 6)).reshape(body_pose.shape[0], -1)  # (N, J_R*3)

        # 3. Hand and Face BBox Estimation
        lhand_bbox_center, lhand_bbox_size, rhand_bbox_center, rhand_bbox_size, face_bbox_center, face_bbox_size = self.box_net(img_feat, body_joint_hm.detach())
        lhand_bbox = self._restore_bbox(lhand_bbox_center, lhand_bbox_size, self.input_hand_shape[1] / self.input_hand_shape[0], 2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space
        rhand_bbox = self._restore_bbox(rhand_bbox_center, rhand_bbox_size, self.input_hand_shape[1] / self.input_hand_shape[0], 2.0).detach()  # xyxy in (cfg.input_body_shape[1], cfg.input_body_shape[0]) space

        # 4. Differentiable Feature-level Hand Crop-Upsample
        # hand_feat: list, [bsx2, c, cfg.output_hm_shape[1]*scale, cfg.output_hm_shape[2]*scale]
        hand_feat = self.hand_roi_net(img_feat, lhand_bbox, rhand_bbox)  # hand_feat: flipped left hand + right hand

        # 5. Hand/Face Regressor
        # hand regressor
        _, hand_joint_img = self.hand_position_net(hand_feat)  # (2N, J_P, 3)
        hand_pose = self.hand_regressor(hand_feat, hand_joint_img.detach())
        hand_pose = rot6d_to_axis_angle(hand_pose.reshape(-1, 6)).reshape(hand_feat.shape[0], -1)  # (2N, J_R*3)

        # restore flipped left hand joint rotations
        batch_size = hand_pose.shape[0] // 2
        lhand_pose = hand_pose[:batch_size, :].reshape(-1, 15, 3)
        lhand_pose = torch.cat((lhand_pose[:, :, 0:1], -lhand_pose[:, :, 1:3]), 2).view(batch_size, -1)
        rhand_pose = hand_pose[batch_size:, :]

        # hand regressor
        expr, jaw_pose = self.face_regressor(expr_token, jaw_pose_token)
        jaw_pose = rot6d_to_axis_angle(jaw_pose)


        # test output
        out = {}
        
        out['smplx_root_pose'] = root_pose
        out['smplx_body_pose'] = body_pose
        out['smplx_lhand_pose'] = lhand_pose
        out['smplx_rhand_pose'] = rhand_pose
        out['smplx_jaw_pose'] = jaw_pose
        out['smplx_shape'] = shape
        out['smplx_expr'] = expr
        

        return out
    
    def _restore_bbox(self, bbox_center, bbox_size, aspect_ratio, extension_ratio):
        bbox = bbox_center.view(-1, 1, 2) + torch.cat((-bbox_size.view(-1, 1, 2) / 2., bbox_size.view(-1, 1, 2) / 2.),
                                                  1)  # xyxy in (cfg.output_hm_shape[2], cfg.output_hm_shape[1]) space
        bbox[:, :, 0] = bbox[:, :, 0] / self.output_hm_shape[2] * self.input_body_shape[1]
        bbox[:, :, 1] = bbox[:, :, 1] / self.output_hm_shape[1] * self.input_body_shape[0]
        bbox = bbox.view(-1, 4)

        # xyxy -> xywh
        bbox[:, 2] = bbox[:, 2] - bbox[:, 0]
        bbox[:, 3] = bbox[:, 3] - bbox[:, 1]

        # aspect ratio preserving bbox
        w = bbox[:, 2]
        h = bbox[:, 3]
        c_x = bbox[:, 0] + w / 2.
        c_y = bbox[:, 1] + h / 2.

        mask1 = w > (aspect_ratio * h)
        mask2 = w < (aspect_ratio * h)
        h[mask1] = w[mask1] / aspect_ratio
        w[mask2] = h[mask2] * aspect_ratio

        bbox[:, 2] = w * extension_ratio
        bbox[:, 3] = h * extension_ratio
        bbox[:, 0] = c_x - bbox[:, 2] / 2.
        bbox[:, 1] = c_y - bbox[:, 3] / 2.

        # xywh -> xyxy
        bbox[:, 2] = bbox[:, 2] + bbox[:, 0]
        bbox[:, 3] = bbox[:, 3] + bbox[:, 1]
        return bbox
