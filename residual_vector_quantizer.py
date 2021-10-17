# Copyright (c) 2020 Phil Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

import torch
from torch import nn
import torch.nn.functional as F


class KMeans:
    def __init__(self, dim, n_c, n_iter=1000, eps=1e-4):
        self.n_c = n_c
        self.n_iter = n_iter
        self.eps = eps
        self.centroids = torch.randn(dim, n_c)
    
    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
    
    def predict(self, X):
        dist = (
            X.pow(2).sum(1, keepdim=True)
            - 2 * X @ self.centroids
            + self.centroids.pow(2).sum(0, keepdim=True)
        )
        _, centroids_ind = (-dist).max(1)
        centroids_ind = centroids_ind.view(*X.shape[:-1])
        return centroids_ind
    
    def train(self, X):
        dtype = X.dtype
        for n in range(self.n_iter):
            dist = (
            X.pow(2).sum(1, keepdim=True)
            - 2 * X @ self.centroids
            + self.centroids.pow(2).sum(0, keepdim=True)
            )
            _, centroids_ind = (-dist).max(1)
            centroids_onehot = F.one_hot(centroids_ind, self.n_c).type(dtype)
            new_centroids = X.transpose(0, 1) @ centroids_onehot
            new_centroids /= centroids_onehot.sum(0)
            if (self.centroids - new_centroids).pow(2).sum().sqrt() < self.eps:
                break
        return True, n
            


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha = (1 - decay))

def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        decay = 0.99,
        commitment = 1.,
        eps = 1e-5,
        n_embed = None,
    ):
        super().__init__()
        n_embed = default(n_embed, codebook_size)

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment

        embed = None
        self.register_buffer('embed', embed)
        self.register_buffer('cluster_size', torch.zeros(n_embed))
        self.register_buffer('embed_avg', None)

    @property
    def codebook(self):
        return self.embed.transpose(0, 1)

    def forward(self, input):
        dtype = input.dtype
        flatten = input.reshape(-1, self.dim)
        
        if self.embed == None:
            kmeans = KMeans(dim=self.dim, n_c=self.n_embed)
            kmeans.train(input)
            self.embed = kmeans.centroids
        
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        if self.training:
            ema_inplace(self.cluster_size, embed_onehot.sum(0), self.decay)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot
            ema_inplace(self.embed_avg, embed_sum, self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.n_embed, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
 
        quantize = input + (quantize - input).detach()
        return quantize, embed_ind


class ResidualVQ(nn.Module):
    """ Follows Algorithm 1. in https://arxiv.org/pdf/2107.03312.pdf """
    def __init__(
        self,
        *,
        num_quantizers,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([VectorQuantizer(**kwargs) for _ in range(num_quantizers)])

    def forward(self, x):
        quantized_out = 0.
        residual = x

        all_indices = []

        for layer in self.layers:
            quantized, indices = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized

            all_indices.append(indices)

        all_indices = map(torch.stack, all_indices)
        return quantized_out, all_indices