# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

# Closest point function + texture sampling
# https://en.wikipedia.org/wiki/Closest_point_method

import torch
import numpy as np
from .barycentric_coordinates import barycentric_coordinates
from tqdm import tqdm
from kaolin.ops.mesh import index_vertices_by_faces, check_sign
from kaolin import _C


class _UnbatchedTriangleDistanceCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, face_vertices):
        num_points = points.shape[0]
        num_faces = face_vertices.shape[0]
        min_dist = torch.zeros((num_points), device=points.device, dtype=points.dtype)
        min_dist_idx = torch.zeros((num_points), device=points.device, dtype=torch.long)
        dist_type = torch.zeros((num_points), device=points.device, dtype=torch.int32)
        _C.metrics.unbatched_triangle_distance_forward_cuda(
            points, face_vertices, min_dist, min_dist_idx, dist_type)
        ctx.save_for_backward(points.contiguous(), face_vertices.contiguous(),
                              min_dist_idx, dist_type)
        ctx.mark_non_differentiable(min_dist_idx, dist_type)
        return min_dist, min_dist_idx, dist_type

    @staticmethod
    def backward(ctx, grad_dist, grad_face_idx, grad_dist_type):
        points, face_vertices, face_idx, dist_type = ctx.saved_tensors
        grad_dist = grad_dist.contiguous()
        grad_points = torch.zeros_like(points)
        grad_face_vertices = torch.zeros_like(face_vertices)
        _C.metrics.unbatched_triangle_distance_backward_cuda(
            grad_dist, points, face_vertices, face_idx, dist_type,
            grad_points, grad_face_vertices)
        return grad_points, grad_face_vertices


def _compute_dot(p1, p2):
    return p1[..., 0] * p2[..., 0] + \
        p1[..., 1] * p2[..., 1] + \
        p1[..., 2] * p2[..., 2]

def _project_edge(vertex, edge, point):
    point_vec = point - vertex
    length = _compute_dot(edge, edge)
    return _compute_dot(point_vec, edge) / length

def _project_plane(vertex, normal, point):
    point_vec = point - vertex
    unit_normal = normal / torch.norm(normal, dim=-1, keepdim=True)
    dist = _compute_dot(point_vec, unit_normal)
    return point - unit_normal * dist.view(-1, 1)

def _is_not_above(vertex, edge, norm, point):
    edge_norm = torch.cross(norm, edge, dim=-1)
    return _compute_dot(edge_norm.view(1, -1, 3),
                        point.view(-1, 1, 3) - vertex.view(1, -1, 3)) <= 0

def _point_at(vertex, edge, proj):
    return vertex + edge * proj.view(-1, 1)


def _unbatched_naive_point_to_mesh_distance(points, face_vertices):
    """
    description of distance type:
        - 0: distance to face
        - 1: distance to vertice 0
        - 2: distance to vertice 1
        - 3: distance to vertice 2
        - 4: distance to edge 0-1
        - 5: distance to edge 1-2
        - 6: distance to edge 2-0
    Args:
        points (torch.Tensor): of shape (num_points, 3).
        faces_vertices (torch.LongTensor): of shape (num_faces, 3, 3).
    Returns:
        (torch.Tensor, torch.LongTensor, torch.IntTensor):
            - distance, of shape (num_points).
            - face_idx, of shape (num_points).
            - distance_type, of shape (num_points).
            - conter P
    """
    num_points = points.shape[0]
    num_faces = face_vertices.shape[0]

    device = points.device
    dtype = points.dtype

    v1 = face_vertices[:, 0]
    v2 = face_vertices[:, 1]
    v3 = face_vertices[:, 2]

    e21 = v2 - v1
    e32 = v3 - v2
    e13 = v1 - v3

    normals = -torch.cross(e21, e13)

    uab = _project_edge(v1.view(1, -1, 3), e21.view(1, -1, 3), points.view(-1, 1, 3))
    ubc = _project_edge(v2.view(1, -1, 3), e32.view(1, -1, 3), points.view(-1, 1, 3))
    uca = _project_edge(v3.view(1, -1, 3), e13.view(1, -1, 3), points.view(-1, 1, 3))

    is_type1 = (uca > 1.) & (uab < 0.)
    is_type2 = (uab > 1.) & (ubc < 0.)
    is_type3 = (ubc > 1.) & (uca < 0.)
    is_type4 = (uab >= 0.) & (uab <= 1.) & _is_not_above(v1, e21, normals, points)
    is_type5 = (ubc >= 0.) & (ubc <= 1.) & _is_not_above(v2, e32, normals, points)
    is_type6 = (uca >= 0.) & (uca <= 1.) & _is_not_above(v3, e13, normals, points)
    is_type0 = ~(is_type1 | is_type2 | is_type3 | is_type4 | is_type5 | is_type6)

    face_idx = torch.zeros(num_points, device=device, dtype=torch.long)
    all_closest_points = torch.zeros((num_points, num_faces, 3), device=device,
                                     dtype=dtype)

    all_type0_idx = torch.where(is_type0)
    all_type1_idx = torch.where(is_type1)
    all_type2_idx = torch.where(is_type2)
    all_type3_idx = torch.where(is_type3)
    all_type4_idx = torch.where(is_type4)
    all_type5_idx = torch.where(is_type5)
    all_type6_idx = torch.where(is_type6)

    all_types = is_type1.int() + is_type2.int() * 2 + is_type3.int() * 3 + \
        is_type4.int() * 4 + is_type5.int() * 5 + is_type6.int() * 6

    all_closest_points[all_type0_idx] = _project_plane(
        v1[all_type0_idx[1]], normals[all_type0_idx[1]], points[all_type0_idx[0]])
    all_closest_points[all_type1_idx] = v1.view(-1, 3)[all_type1_idx[1]]
    all_closest_points[all_type2_idx] = v2.view(-1, 3)[all_type2_idx[1]]
    all_closest_points[all_type3_idx] = v3.view(-1, 3)[all_type3_idx[1]]
    all_closest_points[all_type4_idx] = _point_at(v1[all_type4_idx[1]], e21[all_type4_idx[1]],
                                                  uab[all_type4_idx])
    all_closest_points[all_type5_idx] = _point_at(v2[all_type5_idx[1]], e32[all_type5_idx[1]],
                                                  ubc[all_type5_idx])
    all_closest_points[all_type6_idx] = _point_at(v3[all_type6_idx[1]], e13[all_type6_idx[1]],
                                                  uca[all_type6_idx])
    all_vec = (all_closest_points - points.view(-1, 1, 3))
    all_dist = _compute_dot(all_vec, all_vec)

    _, min_dist_idx = torch.min(all_dist, dim=-1)
    dist_type = all_types[torch.arange(num_points, device=device), min_dist_idx]
    torch.cuda.synchronize()

    # Recompute the shortest distances
    # This reduce the backward pass to the closest faces instead of all faces
    # O(num_points) vs O(num_points * num_faces)
    selected_face_vertices = face_vertices[min_dist_idx]
    v1 = selected_face_vertices[:, 0]
    v2 = selected_face_vertices[:, 1]
    v3 = selected_face_vertices[:, 2]

    e21 = v2 - v1
    e32 = v3 - v2
    e13 = v1 - v3

    normals = -torch.cross(e21, e13)

    uab = _project_edge(v1, e21, points)
    ubc = _project_edge(v2, e32, points)
    uca = _project_edge(v3, e13, points)

    counter_p = torch.zeros((num_points, 3), device=device, dtype=dtype)

    cond = (dist_type == 1)
    counter_p[cond] = v1[cond]

    cond = (dist_type == 2)
    counter_p[cond] = v2[cond]

    cond = (dist_type == 3)
    counter_p[cond] = v3[cond]

    cond = (dist_type == 4)
    counter_p[cond] = _point_at(v1, e21, uab)[cond]

    cond = (dist_type == 5)
    counter_p[cond] = _point_at(v2, e32, ubc)[cond]

    cond = (dist_type == 6)
    counter_p[cond] = _point_at(v3, e13, uca)[cond]

    cond = (dist_type == 0)
    counter_p[cond] = _project_plane(v1, normals, points)[cond]
    min_dist = torch.sum((counter_p - points) ** 2, dim=-1)

    return min_dist, min_dist_idx, dist_type, counter_p


def _find_closest_point(points, face_vertices, cur_face_idx, cur_dist_type):
    """Returns the closest point given a querypoints and meshes.
        points (torch.Tensor): of shape (num_points, 3).
        faces_vertices (torch.LongTensor): of shape (num_faces, 3, 3).
        cur_face_idx (torch.LongTensor): of shape (num_points,).
        cur_dist_type (torch.LongTensor): of shape (num_points,).

    Returns:
        (torch.FloatTensor): counter_p of shape (num_points, 3).
    """
    num_points = points.shape[0]
    device = points.device
    dtype = points.dtype
    selected_face_vertices = face_vertices[cur_face_idx]

    v1 = selected_face_vertices[:, 0]
    v2 = selected_face_vertices[:, 1]
    v3 = selected_face_vertices[:, 2]

    e21 = v2 - v1
    e32 = v3 - v2
    e13 = v1 - v3

    normals = -torch.cross(e21, e13)

    uab = _project_edge(v1, e21, points)
    ubc = _project_edge(v2, e32, points)
    uca = _project_edge(v3, e13, points)

    counter_p = torch.zeros((num_points, 3), device=device, dtype=dtype)

    cond = (cur_dist_type == 1)
    counter_p[cond] = v1[cond]

    cond = (cur_dist_type == 2)
    counter_p[cond] = v2[cond]

    cond = (cur_dist_type == 3)
    counter_p[cond] = v3[cond]

    cond = (cur_dist_type == 4)
    counter_p[cond] = _point_at(v1, e21, uab)[cond]

    cond = (cur_dist_type == 5)
    counter_p[cond] = _point_at(v2, e32, ubc)[cond]

    cond = (cur_dist_type == 6)
    counter_p[cond] = _point_at(v3, e13, uca)[cond]

    cond = (cur_dist_type == 0)
    counter_p[cond] = _project_plane(v1, normals, points)[cond]


    return counter_p

def closest_point(
    V : torch.Tensor, 
    F : torch.Tensor,
    points : torch.Tensor,
    split_size : int = 5*10**3):

    """Returns the closest texture for a set of points.

        V (torch.FloatTensor): mesh vertices of shape [V, 3] 
        F (torch.LongTensor): mesh face indices of shape [F, 3]
        points (torch.FloatTensor): sample locations of shape [N, 3]

    Returns:
        (torch.FloatTensor): distances of shape [N, 1]
        (torch.FloatTensor): projected points of shape [N, 3]
        (torch.FloatTensor): face indices of shape [N, 1]
    """

    V = V.cuda().contiguous()
    F = F.cuda().contiguous()

    mesh = index_vertices_by_faces(V.unsqueeze(0), F).squeeze(0)

    _points = torch.split(points, split_size)

    dists = []
    pts = []
    indices = []
    for _p in _points:
        p = _p.cuda().contiguous()
        sign = check_sign(V.unsqueeze(0), F, p.unsqueeze(0)).squeeze(0)
        dist, hit_tidx, dist_type, hit_pts = _unbatched_naive_point_to_mesh_distance(p, mesh)
        dist = torch.where (sign, -torch.sqrt(dist), torch.sqrt(dist))
        dists.append(dist)
        pts.append(hit_pts)
        indices.append(hit_tidx)

    return torch.cat(dists)[...,None], torch.cat(pts), torch.cat(indices)

def batched_closest_point(
    V : torch.Tensor, 
    F : torch.Tensor,
    points : torch.Tensor):

    """Returns the closest texture for a set of points.

        V (torch.FloatTensor): mesh vertices of shape [B, V, 3] 
        F (torch.LongTensor): mesh face indices of shape [F, 3]
        points (torch.FloatTensor): sample locations of shape [B, N, 3]

    Returns:
        (torch.FloatTensor): distances of shape [B, N, 1]
        (torch.FloatTensor): projected points of shape [B, N, 3]
        (torch.FloatTensor): face indices of shape [B, N, 1]
    """

    V = V.cuda().contiguous()
    F = F.cuda().contiguous()

    batch_size = V.shape[0]
    num_points = V.shape[1]

    dists = []
    pts = []
    indices = []
    weights = []

    sign = check_sign(V, F, points)

    for i in range(batch_size):
        mesh = V[i][F]
        p = points[i]
        dist, hit_tidx, dist_type, hit_pts = _unbatched_naive_point_to_mesh_distance(p, mesh)
        dist = torch.where (sign[i], -torch.sqrt(dist), torch.sqrt(dist))
        hitface = F[hit_tidx.view(-1)] # [ Ns , 3]


        BC = barycentric_coordinates(hit_pts, V[i][hitface[:,0]],
                                    V[i][hitface[:,1]], V[i][hitface[:,2]])

        dists.append(dist)
        pts.append(hit_pts)
        indices.append(hit_tidx)
        weights.append(BC)
    
    return torch.stack(dists)[...,None], torch.stack(pts), torch.stack(indices), torch.stack(weights)


def closest_point_fast(
    V : torch.Tensor, 
    F : torch.Tensor,
    points : torch.Tensor):

    """Returns the closest texture for a set of points.

        V (torch.FloatTensor): mesh vertices of shape [V, 3] 
        F (torch.LongTensor): mesh face indices of shape [F, 3]
        points (torch.FloatTensor): sample locations of shape [N, 3]

    Returns:
        (torch.FloatTensor): signed distances of shape [N, 1]
        (torch.FloatTensor): projected points of shape [N, 3]
        (torch.FloatTensor): face indices of shape [N, ]
    """

    face_vertices =  V[F]
    sign = check_sign(V.unsqueeze(0), F, points.unsqueeze(0)).squeeze(0)

    if points.is_cuda:
        cur_dist, cur_face_idx, cur_dist_type = _UnbatchedTriangleDistanceCuda.apply(
                points, face_vertices)
    else:
        cur_dist, cur_face_idx, cur_dist_type = _unbatched_naive_point_to_mesh_distance(
                points, face_vertices)

    hit_point = _find_closest_point(points, face_vertices, cur_face_idx, cur_dist_type)

    dist = torch.where (sign, -torch.sqrt(cur_dist), torch.sqrt(cur_dist))


    return dist[...,None], hit_point, cur_face_idx


def batched_closest_point_fast(
    V : torch.Tensor, 
    F : torch.Tensor,
    points : torch.Tensor):

    """Returns the closest texture for a set of points.

        V (torch.FloatTensor): mesh vertices of shape [B, V, 3] 
        F (torch.LongTensor): mesh face indices of shape [F, 3]
        points (torch.FloatTensor): sample locations of shape [B, N, 3]

    Returns:
        (torch.FloatTensor): distances of shape [B, N, 1]
        (torch.FloatTensor): projected points of shape [B, N, 3]
        (torch.FloatTensor): face indices of shape [B, N, 1]
    """

    batch_size = V.shape[0]

    dists = []
    indices = []
    weights = []
    pts = []

    for i in range(batch_size):
        cur_dist, hit_point, cur_face_idx = closest_point_fast (V[i], F, points[i])
        hitface = F[cur_face_idx.view(-1)] # [ N , 3]

        dists.append(cur_dist)
        pts.append(hit_point)
        indices.append(cur_face_idx)
        weights.append(barycentric_coordinates(hit_point, V[i][hitface[:,0]],
                                    V[i][hitface[:,1]], V[i][hitface[:,2]]))
    
    return torch.stack(dists, dim=0), torch.stack(pts, dim=0), \
           torch.stack(indices, dim=0), torch.stack(weights, dim=0)