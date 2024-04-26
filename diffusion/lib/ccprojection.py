"""
Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
"""

import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin

class CCProjection(ModelMixin, ConfigMixin):
    def __init__(self, clip_image_encoder):
        super().__init__()

        self.projection = torch.nn.Linear(clip_image_encoder.visual_projection.in_features,
                                          clip_image_encoder.visual_projection.out_features,
                                          bias=False)
        self.projection.load_state_dict(clip_image_encoder.visual_projection.state_dict())
    def forward(self, x):
        return self.projection(x)

