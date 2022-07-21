import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from random import randrange

def dropout_layers(layers, prob_survival):
    if prob_survival == 1:
        return layers

    num_layers = len(layers)
    to_drop = torch.zeros(num_layers).uniform_(0., 1.) > prob_survival

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

def cosine_similarity(model, x, y):
    with torch.no_grad():
        x_embed = model.get_output(x).masked_fill((x==0).unsqueeze(2).repeat(1,1,512), 0.0)
        y_embed = model.get_output(y).masked_fill((y==0).unsqueeze(2).repeat(1,1,512), 0.0)
        x_embed = torch.mean(x_embed, dim=1)
        y_embed = torch.mean(y_embed, dim=1)

    return F.cosine_similarity(x_embed, y_embed, dim=1)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class SpatialGatingUnit(nn.Module):
    def __init__(self, d_in, max_len, act, init_eps):
        super(SpatialGatingUnit, self).__init__()
        self.max_len = max_len

        self.d_out = d_in//2
        # normalization
        self.norm = nn.LayerNorm(self.d_out)
        # activation
        self.act = act
        # spatial projection
        init_eps /= max_len
        w = torch.empty(2*max_len-1)
        b = torch.empty(max_len)
        init.uniform_(w, -init_eps, init_eps)
        init.constant_(b, 1.)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(b)

    def toeplitz_spatial_weight_matrix(self):
        W = F.pad(self.w, (0, self.max_len))
        W = W.repeat(self.max_len,)
        W = W[:-self.max_len]
        W = W.view(-1, self.max_len*3-2)
        return W[:,self.max_len-1:1-self.max_len]

    def forward(self, x):
        x_1, x_2 = x.chunk(2, dim=-1)
        x_2 = self.norm(x_2)
        W = self.toeplitz_spatial_weight_matrix()
        x_2 = F.conv1d(x_2, W.unsqueeze(2), self.b)

        return x_1 * self.act(x_2)

class gMLPBlock(nn.Module):
    def __init__(self, d_in, d_ffn, max_len, act):
        super(gMLPBlock, self).__init__()
        # channel projection
        self.proj_in = nn.Sequential(
            nn.Linear(d_in, d_ffn),
            nn.GELU()
        )
        # spatial gating unit
        self.sgu = SpatialGatingUnit(d_ffn, max_len, act, 1e-3)
        # channel projection
        self.proj_out = nn.Linear(d_ffn//2, d_in)

    def forward(self, x):
        x = self.proj_in(x)
        x = self.sgu(x)
        x = self.proj_out(x)

        return x

class HierarchicalLabelling(nn.Module):
    def __init__(self, d_in, *n_dims):
        self.to_logits = nn.ModuleList([nn.Linear(d_in, n_dim) for n_dim in n_dims])

    def set_hierarchy(self, *tensors):
        for i, tensor in enumerate(tensors):
            self.register_buffer("{}_to_{}".format(i, i+1), tensor)

    def forward(self, x):
        output = []
        for i, fn in enumerate(self.to_logits):
            out = fn(x) + torch.matmul(out, getattr(self, "{}_to_{}".format(i-1, i))) if i > 0 else fn(x)
            output.append(out)

        return output

class gMLP(nn.Module):
    def __init__(self, n_tokens, d_in, d_ffn, max_len, n_layers, act):
        super(gMLP, self).__init__()
        self.embedding = nn.Embedding(n_tokens, d_in)
        self.max_len = max_len
        layers = nn.ModuleList([Residual(PreNorm(d_in, gMLPBlock(d_in, d_ffn, max_len, act))) for i in range(n_layers)])
        self.layers = nn.Sequential(*layers)

        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, n_tokens)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.layers(x)
        self.output_embedding = x
        return self.to_logits(x)

    def get_output(self, x):
        x = self.embedding(x)
        x = self.layers(x)
        return x
    
    # N x L x d
    # weights: L+1
    def embed(self, x, weights):
        x = self.embedding(x)
        embedding = weights[0] * x
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            embedding = embedding + weights[i+1] * x
            
        return embedding

class MultigMLP(nn.Module):
    def __init__(self, n_tokens, d_in, d_ffn, max_len, n_layers, act, n_species):
        super(MultigMLP, self).__init__()
        self.embedding = nn.Embedding(n_tokens, d_in)
        self.max_len = max_len
        layers = nn.ModuleList([Residual(PreNorm(d_in, gMLPBlock(d_in, d_ffn, max_len, act))) for i in range(n_layers)])
        self.layers = nn.Sequential(*layers)

        #ここが線形のdecoder部分(固定して使いたいやつ)
        self.to_logits = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, n_tokens)
        )

        #ここが線形のdecoder部分(固定して使いたいやつ)：こっちは菌種分類の方か
        self.to_species = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, n_species)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.layers(x)

        return self.to_logits(x), self.to_species(x[:,0])

    def get_output(self, x):
        #mask = x == 0
        x = self.embedding(x)
        x = self.layers(x)
        #x = self.to_logits[0](x)
        #x = x.masked_fill(mask.unsqueeze(2).repeat(1,1,x.size(2)), 0)
        return x[:,1:]
