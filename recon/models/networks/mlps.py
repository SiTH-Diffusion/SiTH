# The code is adapted from https://github.com/NVIDIAGameWorks/kaolin-wisp/blob/main/wisp/models/decoders/basic_decoders.py

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MLP(nn.Module):
    """Super basic but super useful MLP class.
    """
    def __init__(self, 
        input_dim, 
        output_dim, 
        activation = torch.relu,
        bias = True,
        layer = nn.Linear,
        num_layers = 4, 
        hidden_dim = 128, 
        skip       = [2]
    ):
        """Initialize the MLP.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim        
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip
        if self.skip is None:
            self.skip = []
        
        self.make()

    def make(self):
        """Builds the actual MLP.
        """
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.input_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)

    def forward(self, x, return_h=False, sigmoid=False):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:
                h = torch.cat([x, h], dim=-1)
                h = self.activation(l(h))
            else:
                h = self.activation(l(h))
        
        out = self.lout(h)
        
        if sigmoid:
            out = torch.sigmoid(out) 

        if return_h:
            return out, h
        else:
            return out



class Conditional_MLP(nn.Module):
    """Super basic but super useful MLP class.
    """
    def __init__(self, 
        input_dim, 
        cond_dim,
        output_dim, 
        activation = torch.relu,
        bias = True,
        layer = nn.Linear,
        num_layers = 4, 
        hidden_dim = 128, 
        skip       = [2]
    ):
        """Initialize the MLP.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.output_dim = output_dim        
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip = skip
        if self.skip is None:
            self.skip = []
        
        self.make()

    def make(self):
        """Builds the actual MLP.
        """
        layers = []
        for i in range(self.num_layers):
            if i == 0: 
                layers.append(self.layer(self.input_dim, self.hidden_dim, bias=self.bias))
            elif i in self.skip:
                layers.append(self.layer(self.hidden_dim+self.cond_dim, self.hidden_dim, bias=self.bias))
            else:
                layers.append(self.layer(self.hidden_dim, self.hidden_dim, bias=self.bias))
        self.layers = nn.ModuleList(layers)
        self.lout = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)
        #self.lh = self.layer(self.hidden_dim, self.output_dim, bias=self.bias)

    def forward(self, x, c, return_h=False, sigmoid=False):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """
        N = x.shape[0]

        for i, l in enumerate(self.layers):
            if i == 0:
                h = self.activation(l(x))
            elif i in self.skip:
                h = torch.cat([h, c], dim=-1)
                h = self.activation(l(h))
            else:
                h = self.activation(l(h))
        
        out = self.lout(h)
        if sigmoid:
            out = torch.sigmoid(out) 

        if return_h:
            return out, h
        else:
            return out



# Transformer implementation based on ViT
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., selfatt=True, kv_dim=None):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        if selfatt:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, z=None):
        if z is None:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
        else:
            q = self.to_q(x)
            k, v = self.to_kv(z).chunk(2, dim=-1)
            qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., selfatt=True, kv_dim=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout, selfatt=selfatt, kv_dim=kv_dim)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, z=None):
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = ff(x) + x
        return x


class Transformer_decoder(nn.Module):
    """Super basic but super useful MLP class.
    """
    def __init__(self, 
        input_dim, 
        cond_dim,
        output_dim, 
        activation = torch.relu,
        bias = True,
        layer = nn.Linear,
        num_layers = 2, 
        hidden_dim = 128, 
    ):
        """Initialize the MLP.

        Args:
            input_dim (int): Input dimension of the MLP.
            output_dim (int): Output dimension of the MLP.
            activation (function): The activation function to use.
            bias (bool): If True, use bias.
            layer (nn.Module): The MLP layer module to use.
            num_layers (int): The number of hidden layers in the MLP.
            hidden_dim (int): The hidden dimension of the MLP.
            skip (List[int]): List of layer indices where the input dimension is concatenated.

        Returns:
            (void): Initializes the class.
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.output_dim = output_dim        
        self.activation = activation
        self.bias = bias
        self.layer = layer
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        self.make()

    def make(self):
        """Builds the actual MLP.
        """

        self.input_mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))
        
        self.output_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim))
        
        self.z_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim))

        self.transformer = Transformer(self.hidden_dim, depth=self.num_layers, heads=8, dim_head=self.hidden_dim // 8,
                                       mlp_dim=self.hidden_dim * 2, selfatt=False, kv_dim=self.hidden_dim)


    def forward(self, x, z, return_h=False, sigmoid=False):
        """Run the MLP!

        Args:
            x (torch.FloatTensor): Some tensor of shape [batch, ..., input_dim]
            return_h (bool): If True, also returns the last hidden layer.

        Returns:
            (torch.FloatTensor, (optional) torch.FloatTensor):
                - The output tensor of shape [batch, ..., output_dim]
                - The last hidden layer of shape [batch, ..., hidden_dim]
        """

        B, N, L, Dz = z.shape

        queries = self.input_mlp(x.view(-1, 1, self.input_dim))

        #queries = self.input_mlp(x.view(-1, 1, Dx))
        z = self.z_mlp(z.view(-1, L, Dz))

        h = self.transformer(queries, z)

        output = self.output_mlp(h)

        if sigmoid:
            output = torch.sigmoid(output) 

        if return_h:
            return output.view(B, N, -1), h.view(B, N, -1)
        else:
            return output.view(B, N, -1)




def get_activation_class(activation_type):
    """Utility function to return an activation function class based on the string description.

    Args:
        activation_type (str): The name for the activation function.
    
    Returns:
        (Function): The activation function to be used. 
    """
    if activation_type == 'relu':
        return torch.relu
    elif activation_type == 'sin':
        return torch.sin
    elif activation_type == 'softplus':
        return torch.nn.functional.softplus
    elif activation_type == 'lrelu':
        return torch.nn.functional.leaky_relu
    else:
        assert False and "activation type does not exist"
