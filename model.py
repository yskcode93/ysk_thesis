# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# basic
import math

from random import randrange
# gated MLP
from g_mlp_pytorch import gMLP
# transformer
from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder, RecurrentEncoderBuilder, RecurrentDecoderBuilder
from fast_transformers.masking import LengthMask, TriangularCausalMask

# this skips layers with the given probability
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

class Converter(gMLP):
    def __init__(self, dim, depth, seq_len, num_tokens, ff_mult, heads, circulant_matrix, act, prob_survival,\
        n_species, pretrain=True, **kwargs):
        super(Converter, self).__init__(
            num_tokens=num_tokens,
            dim=dim,
            depth=depth,
            seq_len=seq_len,
            heads=heads,
            ff_mult=ff_mult,
            attn_dim=None,
            prob_survival=prob_survival,
            causal=False,
            circulant_matrix=circulant_matrix,
            shift_tokens=0,
            act=act
        )

        # species embedding
        if pretrain:
            self.register_buffer("species_embedding", kwargs["species_embedding"])
        else:
            self.species_embedding = nn.Embedding(n_species, dim)
        # pretrain mode
        self.pretrain = pretrain
        
    def forward(self, x_e, x, sp):
        if self.pretrain:
            x = x_e + self.to_embed(x) + self.species_embedding[sp]
        else:
            x = self.to_embed(x) + self.species_embedding(sp)

        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        out = nn.Sequential(*layers)(x)
        
        return self.to_logits(out)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# Transformer based converter
class TransformerConverter(nn.Module):
    def __init__(self, n_tokens, seq_len, n_layers, n_heads, query_dimensions, value_dimensions, feed_forward_dimensions, attention_type,\
            n_species, pretrain = True, **kwargs
        ):
        super(TransformerConverter, self).__init__()

        self.layers = TransformerEncoderBuilder.from_kwargs(
            n_layers = n_layers,
            n_heads = n_heads,
            query_dimensions = query_dimensions,
            value_dimensions = value_dimensions,
            feed_forward_dimensions = feed_forward_dimensions,
            attention_type = attention_type
        ).get()

        dim = n_heads*query_dimensions
        self.positional_encoding = PositionalEncoding(dim, seq_len)
        self.codon_embedding = nn.Embedding(n_tokens, dim)
        self.to_logits = nn.Linear(dim, n_tokens)

        if pretrain:
            self.register_buffer("species_embedding", kwargs["species_embedding"])
        else:
            self.species_embedding = nn.Embedding(n_species, dim)

        self.pretrain = pretrain

    def forward(self, x_e, x, sp):
        x = self.codon_embedding(x)
        x = self.positional_encoding(x)

        if self.pretrain:
            x = x + x_e
            sp = self.species_embedding[sp]
        else:
            sp = self.species_embedding(sp)

        x = torch.cat((sp, x), dim=1) # N x (L+1)
        x = self.layers(x)
        
        return self.to_logits(x)[:, 1:] # discard first

# Autoregressive Converter
"""
class AutoregressiveConverter(nn.Module):
    def __init__(self, n_tokens, seq_len, n_layers, n_heads, query_dimensions, value_dimensions, feed_forward_dimensions, attention_type,\
            n_species, pretrain = True, **kwargs
        ):
        super(AutoregressiveConverter, self).__init__()

        self.encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers = n_layers,
            n_heads = n_heads,
            query_dimensions = query_dimensions,
            value_dimensions = value_dimensions,
            feed_forward_dimensions = feed_forward_dimensions,
            attention_type = attention_type
        ).get()

        self.decoder = TransformerDecoderBuilder.from_kwargs(
            n_layers = n_layers,
            n_heads = n_heads,
            query_dimensions = query_dimensions,
            value_dimensions = value_dimensions,
            feed_forward_dimensions = feed_forward_dimensions,
            self_attention_type = attention_type,
            cross_attention_type = attention_type
        ).get()

        dim = n_heads*query_dimensions
        self.positional_encoding = PositionalEncoding(dim, seq_len)
        self.codon_embedding = nn.Embedding(n_tokens, dim)
        self.to_logits = nn.Linear(dim, n_tokens)

        if pretrain:
            self.register_buffer("species_embedding", kwargs["species_embedding"])
        else:
            self.species_embedding = nn.Embedding(n_species, dim)

        self.pretrain = pretrain

    def forward(self, x_e, x, sp, y=None):
        N, max_len = x.size(0), x.size(1)
        x = self.codon_embedding(x)
        x = self.positional_encoding(x)
        y = self.codon_embedding(y)
        y = self.positional_encoding(y)

        if self.pretrain:
            x = x + x_e
            sp = self.species_embedding[sp]
        else:
            sp = self.species_embedding(sp)

        # start token is species token
        y = torch.cat((sp, y), dim=1)
        # encode source sequences
        memory = self.encoder(x)
        # decode
        causal_mask = TriangularCausalMask(max_len+1, device = x.device)
        out = self.decoder(y, memory, causal_mask)
        
        return self.to_logits(out[:, :-1]) # discard first
    
    @torch.no_grad
    def infer_greedily(self, x_e, x, sp, max_len, eos_tokens=[]):
        # this is inference
        self.eval()
        # generate
        x = self.codon_embedding(x)
        x = self.positional_encoding(x)

        if self.pretrain:
            x = x + x_e
            sp = self.species_embedding[sp]
        else:
            sp = self.species_embedding(sp)

        # encode source sequences
        memory = self.encoder(x)

        # sos token is species token
        preds = torch.empty(x.size(0), 0, device=x.device)
        for i in range(max_len):
            causal_mask = TriangularCausalMask(i+1, device=x.device)
            out = self.decoder(sp, memory, causal_mask)
            out = self.to_logits(out[:, -1:])
            pred = torch.argmax(out, dim=2)
            preds = torch.cat((preds, pred), dim=1)
"""
# Autoregressive Converter
class AutoregressiveConverter(nn.Module):
    def __init__(self, n_tokens, seq_len, n_layers, n_heads, query_dimensions, value_dimensions, feed_forward_dimensions, attention_type,\
            n_species, pretrain = True, **kwargs
        ):
        super(AutoregressiveConverter, self).__init__()

        dim = n_heads*query_dimensions

        self.transformer = nn.Transformer(
            dim,
            n_heads,
            n_layers,
            n_layers,
            feed_forward_dimensions
        )

        self.positional_encoding = PositionalEncoding(dim, seq_len)
        self.x_embedding = nn.Embedding(n_tokens, dim)
        self.y_embedding = nn.Embedding(n_tokens, dim)
        self.to_logits = nn.Linear(dim, n_tokens)

        if pretrain:
            self.register_buffer("species_embedding", kwargs["species_embedding"])
        else:
            self.species_embedding = nn.Embedding(n_species, dim)

        self.pretrain = pretrain

    def forward(self, x_e, x, sp, y):
        x_length_mask = x == 0
        y_length_mask = y == 0
        y_mask = self.transformer.generate_square_subsequent_mask(y.size(1)).to(y.device)
        
        x = self.x_embedding(x)
        x = self.positional_encoding(x)

        if self.pretrain:
            x = x + x_e
            sp = self.species_embedding[sp]
        else:
            sp = self.species_embedding(sp)

        y = self.y_embedding(y)
        # start token is species token
        y = torch.cat((sp, y[:,:-1]), dim=1)
        y = self.positional_encoding(y)

        out = self.transformer(
            x.permute(1,0,2),
            y.permute(1,0,2),
            src_key_padding_mask=x_length_mask,
            tgt_key_padding_mask=y_length_mask,
            tgt_mask=y_mask
        )
        out = self.to_logits(out)

        return out.permute(1,0,2)

    def infer_greedily(self, x_e, x, sp, max_len, eos_tokens=[]):
        # this is inference
        self.eval()
        # mask
        x_length_mask = x == 0
        # generate
        x = self.x_embedding(x)
        x = self.positional_encoding(x)

        if self.pretrain:
            x = x + x_e
            sp = self.species_embedding[sp]
        else:
            sp = self.species_embedding(sp)

        results = torch.zeros(x.size(0), max_len).long().to(x.device)

        # first prediction
        sp = self.positional_encoding(sp)
        out = self.transformer(
            x.permute(1,0,2),
            sp.permute(1,0,2),
            src_key_padding_mask=x_length_mask
        )
        out = self.to_logits(out).permute(1,0,2)
        preds = torch.argmax(out, dim=2)
        # downstream prediction
        for i in range(1, max_len):
            y = torch.cat((sp, self.y_embedding(preds)), dim=1)
            y = self.positional_encoding(y)
            out = self.transformer(
                x.permute(1,0,2),
                y.permute(1,0,2),
                src_key_padding_mask=x_length_mask
            )
            out = self.to_logits(out).permute(1,0,2)
            pred = torch.argmax(out, dim=2)[:,-1:]

            preds = torch.cat((preds, pred), dim=1)

            # if prediction is termination, remove it from preds
            terminated = torch.any(torch.cat([pred==i for i in eos_tokens], dim=1), dim=1)
            where = torch.zeros(results.size(0)).bool().to(results.device)
            where[torch.all(results==0, dim=1)] = terminated
            results[where, :preds.size(1)] = preds[terminated].clone()

            preds = preds[~terminated]
            sp = sp[~terminated]
            x = x[~terminated]
            x_length_mask = x_length_mask[~terminated]

            if preds.size(0)==0:
                break

        return results

    def beam_search(self, x_e, x, sp, max_len, k, eos_tokens=[]):
        # this is inference
        self.eval()
        # mask
        x_length_mask = x == 0
        # generate
        x = self.x_embedding(x)
        x = self.positional_encoding(x)

        if self.pretrain:
            x = x + x_e
            sp = self.species_embedding[sp]
        else:
            sp = self.species_embedding(sp)

        # predict k start codons per sequence
        # first prediction
        sp = self.positional_encoding(sp)
        out = self.transformer(
            x.permute(1,0,2),
            sp.permute(1,0,2),
            src_key_padding_mask=x_length_mask
        )
        out = self.to_logits(out).permute(1,0,2)
        log_lik, preds = torch.topk(F.log_softmax(out[:,-1], dim=1), k)
        preds = preds.view(-1, 1)
        scores = log_lik.view(-1, 1)

        # downstream prediction
        for i in range(1, max_len):
            y = torch.cat((sp.repeat(k, 1, 1), self.y_embedding(preds)), dim=1)
            y = self.positional_encoding(y)
            out = self.transformer(
                x.permute(1,0,2).repeat(1, k, 1),
                y.permute(1,0,2),
                src_key_padding_mask=x_length_mask.repeat(k, 1)
            )
            out = self.to_logits(out).permute(1,0,2)
            log_lik, cands = torch.topk(F.log_softmax(out[:,-1], dim=1), k)
            log_lik = (log_lik + scores.expand(-1, k) * i) / (i+1)
            # if terminated, score no longer changes and predictions are set to padding token
            terminated = torch.any(torch.cat([preds==i for i in eos_tokens], dim=1), dim=1)
            log_lik[terminated] = scores.expand(-1, k)[terminated].clone()
            cands[terminated] = 0
            # if all sequences terminated, then get out of the loop
            if torch.all(terminated).item():
                break

            cands = cands.view(-1, k)
            scores, pred = torch.topk(log_lik.view(-1, k*k), k)
            pred = pred.view(-1, 1)
            scores = scores.view(-1, 1)

            origin = pred.squeeze() // k + torch.arange(0, x.size(0)*k).to(x.device) // k * k

            preds = torch.cat((preds[origin], cands[torch.arange(x.size(0)*k).to(x.device), (pred%k).squeeze()].unsqueeze(1)), dim=1)

        return preds

# for pretrain
class SPgMLP(gMLP):
    def __init__(
        self,
        *,
        num_tokens = None,
        dim,
        depth,
        seq_len,
        heads = 1,
        ff_mult = 4,
        attn_dim = None,
        prob_survival = 1.,
        causal = False,
        circulant_matrix = False,
        shift_tokens = 0,
        act = nn.Identity(),
        n_species = 1
    ):
        super(SPgMLP, self).__init__(
            num_tokens=num_tokens,
            dim=dim,
            depth=depth,
            seq_len=seq_len,
            heads=heads,
            ff_mult=ff_mult,
            attn_dim=attn_dim,
            prob_survival=prob_survival,
            causal=causal,
            circulant_matrix=circulant_matrix,
            shift_tokens=shift_tokens,
            act=act
        )

        self.species_embedding = nn.Embedding(n_species, dim)

    def get_output(self, x, label):
        x = self.to_embed(x) + self.species_embedding(label)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        out = nn.Sequential(*layers)(x)

        return out

    def forward(self, x, label):
        x = self.to_embed(x) + self.species_embedding(label)
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        out = nn.Sequential(*layers)(x)
        return self.to_logits(out)