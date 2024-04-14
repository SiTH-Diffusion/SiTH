# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch

def per_vertex_normals(
    V : torch.Tensor,
    F : torch.Tensor):
    """Compute normals per face.
    
    Args:
        V (torch.FloatTensor): Vertices of shape [V, 3]
        F (torch.LongTensor): Faces of shape [F, 3]
    
    Returns:
        (torch.FloatTensor): Normals of shape [F, 3]
    """
    verts_normals = torch.zeros_like(V)
    mesh = V[F]

    faces_normals = torch.cross(
        mesh[:, 2] - mesh[:, 1],
        mesh[:, 0] - mesh[:, 1],
        dim=1,
    )

    verts_normals.index_add_(0, F[:, 0], faces_normals)
    verts_normals.index_add_(0, F[:, 1], faces_normals)
    verts_normals.index_add_(0, F[:, 2], faces_normals)
    
    return torch.nn.functional.normalize(
        verts_normals, eps=1e-6, dim=1
    )