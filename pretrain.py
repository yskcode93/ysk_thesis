import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils.rnn import pad_sequence
# apex pytorch extension
from apex.parallel import DistributedDataParallel as DDP_APEX
from apex import amp

import numpy as np

import re
import math
from functools import reduce
from time import time

from mlm_pytorch import MLM

from g_mlp_pytorch import gMLP

from .data_loader import GeneBatchLoader, sanitize

from tqdm import tqdm

from .database import fetchall

from random import randrange

@torch.no_grad()
def get_context_embedding(model, input, label, mask_idx, pad_idx, device=0):
    context_embedding = []
    model = model.to(device)

    for inp in tqdm(torch.unbind(input, dim=0)):
        inp = inp.to(device)
        length = torch.sum(inp != pad_idx).item()
        # create masked input
        inp_masked = inp.clone().unsqueeze(0).repeat(length, 1)
        mask = torch.eye(length, inp.size(0)).bool().to(device)
        inp_masked = inp_masked.masked_fill(mask, mask_idx)
        # context embedding
        embedding = model.get_output(inp_masked, torch.tensor([[label]]*length).to(device))
        context_embedding.append(embedding[mask])

    return pad_sequence(context_embedding, batch_first=True)

@torch.no_grad()
def get_embedding(model, input, label, device=0):
    model = model.to(device)
    embedding = model.get_output(input.to(device), torch.tensor([[label]]*input.size(0)).to(device))

    return embedding

def beam_search(model, vocab, src, tgt_sp, k, device=0):
    model.eval()
    model.to(device)
    
    for s in torch.unbind(src):
        length = torch.sum(s>0).item()
        s_masked = s.clone().unsqueeze(0).repeat(length, 1)
        sp = torch.tensor([[tgt_sp]]*length).expand_as(s_masked)
        s_masked[torch.arange(length), torch.arange(length)] = vocab.index("<msk>")

        sp.to(device)
        s_masked.to(device)

        with torch.no_grad():
            out = model(s_masked, sp)
            neglik = - torch.log(torch.max(torch.softmax(out, dim=2), dim=2)[0])
            pred = torch.argmax(out, dim=2)
            pred[masked==0] = 0


def process_within(z, c, class_idx):
    idx = c==class_idx
    z_class = z[idx]
    centroid = torch.mean(z_class, dim=0)
    var_within = (z_class - centroid) ** 2.0
    var_within = torch.sum(var_within)

    return centroid, var_within

def process_between(centroids, var_withins, counts, class_idx):
    centroid = centroids[class_idx]
    centroids = centroids[class_idx+1:]
    var_within = var_withins[class_idx]
    var_withins = var_withins[class_idx+1:]
    global_means_pair = (centroid + centroids) * 0.5
    between_centroid = (centroid - global_means_pair) ** 2
    between_centroids = (centroids - global_means_pair) ** 2
    var_within_pair = var_within + var_withins
    count = counts[class_idx]
    counts = counts[class_idx+1:]
    between_pair = between_centroid * count + between_centroids * counts.reshape(-1, 1)
    counts_pair = count + counts

    for between, within, pair_count, centeroid\
        in zip(torch.unbind(between_pair), torch.unbind(var_within_pair), \
        torch.unbind(counts_pair), torch.unbind(centroids)):

        d1 = 1.0
        d2 = pair_count - 2.0
        x = between / (between + within)
        x = torch.clip(x, 1.0e-37, 1. - 1.0e-5)   

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

class FStaticLoss(nn.Module):
    def __init__(self, eps_within, eps_between):
        super(FStaticLoss, self).__init__()
        self.eps_within = eps_within
        self.eps_between = eps_between
    
    def forward(z, c):
        c = c.view(-1)
        z = z.view(-1, z.size(2))
        cids, counts = torch.unique(c, return_counts=True)
        centroids = []
        var_withins = []

        for cid in torch.unbind(cids):
            centroid, var_within = process_class(z, c, cid)
            centroids += centroid.unsqueeze(0)
            var_withins += var_within.unsqueeze(0)

        centroids = torch.cat(centroids)
        var_withins = torch.cat(var_withins)

class L1Loss(nn.Module):
    def __init__(self, numel):
        super().__init__()
        self.numel = numel

    def forward(self, x):
        params = []
        for p in x.parameters():
            if p.numel() == self.numel:
                params.append(p.unsqueeze(0))

        return F.l1_loss(torch.cat(params, dim=0))

# helper functions
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

def mlm(gpu, world_size, model, vocab, mask_idx, pad_idx, epoch, batch_size, max_length, gbl_config=None, log="./mlm.log", checkpoint_path="./mlm.pt", use_apex=True):
    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    model.to(gpu)

    # hyperparameters
    lr = 1e-4 * batch_size / 256 #0.0008409613225580023#0.00019596231686927837
    beta_1 = 0.9#0.7556703488111914#0.824373329119272
    beta_2 = 0.999#0.44257688145808977#0.6625656161928292
    lamb = 0.01#0.01354395257963608#0.00021950925754120462

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(beta_1, beta_2), weight_decay=lamb)
    # optimized by apex
    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # restore
    pt = torch.load("./multisp_72.pt", map_location=lambda storage, loc: storage)
    model.load_state_dict(pt["model"])
    optimizer.load_state_dict(pt["optimizer"])
    amp.load_state_dict(pt["amp"])
    del pt

    # construct DDP model
    ddp_model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu])
    # define loss function
    trainer = MLM(
        ddp_model,
        mask_token_id = mask_idx,          # the token id reserved for masking
        pad_token_id = pad_idx,           # the token id for padding
        mask_prob = 0.15,           # masking probability for masked language modeling
        replace_prob = 0.80,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
        num_tokens = 65,
        random_token_prob = 0.0,
        mask_ignore_token_ids = []  # other tokens to exclude from masking, include the [cls] and [sep] here
    ).cuda()

    if rank==0:
        with open(log, "w") as f:
            pass

    # Without Gene Batch Loader
    # Use Gene Batch Loader when total gene count is too large
    if gbl_config is None:
        sids = [
            22096, 15376, 22118, 22146, 8415, 21918, 20123, 452, 18655, 6750, 17659, 421, 22191, 21978, 12722, 17400,\
            15093, 20120, 20313, 20114, 22204, 19272, 17982, 19601, 21259, 22091, 1375, 10427, 18739, 18441, 22200, 22201, 22202, 22203
        ]
        sql = """
            SELECT strain_id, seq_nc, length_nc FROM gene
            WHERE strain_id IN ({})
        """.format(",".join(["%s"]*len(sids)))

        rows = fetchall(sql, sids)
        X = []
        label = []
        for sid, seq, length in rows:
            if length//3 <= max_length:
                seq = sanitize(seq)
                seq = re.split('(...)',seq)[1:-1:2]
                seq = list(map(lambda x: vocab.index(x), seq))
                seq = torch.tensor(seq)
                X.append(seq)
                label.append(sids.index(sid))

        X = pad_sequence(X, batch_first=True)
        label = torch.tensor(label).unsqueeze(1)

        if X.size(1) < max_length:
            X = torch.cat((X, torch.zeros(X.size(0), max_length-X.size(1)).long()), dim=1)

        dataset = TensorDataset(X, label)

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank
        )
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=sampler,
        )

        scheduler = OneCycleLR(
            optimizer,
            lr,
            total_steps=epoch*len(loader),
            pct_start=0.01,
            anneal_strategy='linear'
        )

        for i in range(epoch):
            loss_avg = 0

            # different ordering
            sampler.set_epoch(i)

            # start time
            start = time()

            for x, label in loader:
                x = x.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                label = label.expand_as(x) # change shape into (N, L)
                optimizer.zero_grad()
                loss = trainer(x, label=label)

                if use_apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                scheduler.step()
                loss_avg += loss.item()

            # end time
            end = time()

            loss_avg /= len(loader)

            if rank == 0:
                with open(log, "a") as f:
                    f.write("EPOCH({:0=3}%) LOSS: MLM={:.4f} lr={:.4f}(1e-4) ELAPSED TIME: {:3.1f}s\n".format(int((i+1)*100/epoch), loss_avg, scheduler.get_last_lr()[0]*1e4, end-start))

            if rank==0 and (i+1) % 100 == 0:
                checkpoint = {
                    'model': ddp_model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'amp': amp.state_dict()
                } if use_apex else {
                    'model': ddp_model.module.state_dict()
                }

                torch.save(checkpoint, checkpoint_path)
    else:
        for i in range(epoch):
            # gene batch loader
            gbl = GeneBatchLoader(**gbl_config)

            for j, rows in enumerate(gbl):
                X = []
                for last_id, seq, length in rows:
                    if length//3 <= max_length:
                        seq = sanitize(seq)
                        seq = re.split('(...)',seq)[1:-1:2]
                        seq = list(map(lambda x: vocab.index(x), seq))
                        seq = torch.tensor(seq)
                        X.append(seq)

                gbl.last_id = last_id
                        
                X = pad_sequence(X)
                if X.size(0) < max_length:
                    X = torch.cat((X, torch.zeros(max_length-X.size(0), X.size(1)).long()), dim=0)
                
                sampler = DistributedSampler(
                    X.T,
                    num_replicas=world_size,
                    rank=rank
                )
                loader = torch.utils.data.DataLoader(
                    dataset=X.T,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    sampler=sampler,
                )
                
                # different ordering for each epoch
                sampler.set_epoch(i)

                for x in loader:
                    x = x.cuda(non_blocking=True)
                    optimizer.zero_grad()
                    loss = trainer(x)

                    if use_apex:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    optimizer.step()

                if rank == 0:
                    with open(log, "a") as f:
                        f.write("RANK({:0=2}) EPOCH({:0=3}%) STEP({:0=3}%) LOSS: MLM={:.4f}\n".format(rank, int((i+1)*100/epoch), int((j+1)*100/len(gbl)), loss.item()))

            if rank==0 and (i+1) % 5 == 0:
                checkpoint = {
                    'model': ddp_model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict()
                } if use_apex else {
                    'model': ddp_model.module.state_dict()
                }

                torch.save(checkpoint, checkpoint_path)

    dist.destroy_process_group()

def mlm_with_species_embedding(gpu, world_size, model, vocab, mask_idx, pad_idx, epoch, batch_size, max_length, log="./mlm.log", checkpoint_path="./mlm.pt", use_apex=True):
    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, eps=1e-6)
    # optimized by apex
    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    # construct DDP model
    ddp_model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu])
    # define loss function
    trainer = MLM(
        ddp_model,
        mask_token_id = mask_idx,          # the token id reserved for masking
        pad_token_id = pad_idx,           # the token id for padding
        mask_prob = 0.15,           # masking probability for masked language modeling
        replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
        mask_ignore_token_ids = []  # other tokens to exclude from masking, include the [cls] and [sep] here
    ).cuda()

    if rank==0:
        with open(log, "w") as f:
            pass

    # for getting training data
    sql = """
        SELECT g.id, g.seq_nc, g.length_nc, s.species_id
        FROM (SELECT id, seq_nc, length_nc, strain_id FROM gene WHERE id > %s ORDER BY id FETCH FIRST %s ROWS ONLY) AS g
        INNER JOIN strain AS s ON g.strain_id = s.id;
    """
    count = 83833710
    gene_batch_size = 400000

    # forward pass
    for i in range(epoch):
        # gene batch loader
        gbl = GeneBatchLoader(sql, gene_batch_size, count)

        for j, rows in enumerate(gbl):
            X = []
            label = []
            for last_id, seq, length, species_id in rows:
                if length//3 <= max_length:
                    seq = sanitize(seq)
                    seq = re.split('(...)',seq)[1:-1:2]
                    seq = list(map(lambda x: vocab.index(x), seq))
                    seq = torch.tensor(seq)
                    X.append(seq)
                    label.append(species_id)
            gbl.last_id = last_id
            
            X = pad_sequence(X)
            label = torch.tensor(label).unsqueeze(1)
            if X.size(0) < max_length:
                X = torch.cat((X, torch.zeros(max_length-X.size(0), X.size(1)).long()), dim=0)
            
            dataset = TensorDataset(X.T, label)

            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank
            )
            loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=sampler,
            )

            for x, label in loader:
                x = x.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                label = label.expand_as(x) # change shape into (N, L)
                optimizer.zero_grad()
                loss = trainer(x, label=label)

                if use_apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()

            if rank == 0:
                with open(log, "a") as f:
                    f.write("RANK({:0=2}) EPOCH({:0=3}%) STEP({:0=3}%) LOSS: MLM={:.4f}\n".format(rank, int((i+1)*100/epoch), int((j+1)*100/len(gbl)), loss.item()))

        if rank==0 and (i+1) % 5 == 0:
            checkpoint = {
                'model': ddp_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict()
            } if use_apex else {
                'model': ddp_model.module.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            torch.save(checkpoint, checkpoint_path)

    dist.destroy_process_group()

def mlm_xlm(gpu, world_size, model, vocab, mask_idx, pad_idx, epoch, batch_size, max_length, log="./mlm_xlm.log", checkpoint_path="./mlm_xlm.pt"):
    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    model.to(gpu)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, eps=1e-6)
    # optimized by apex
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    # construct DDP model
    ddp_model = DDP(model)
    # define loss function
    trainer = MLM_XLM(
        ddp_model,
        mask_token_id = mask_idx,          # the token id reserved for masking
        pad_token_id = pad_idx,           # the token id for padding
        mask_prob = 0.15,           # masking probability for masked language modeling
        replace_prob = 0.90,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
        mask_ignore_token_ids = []  # other tokens to exclude from masking, include the [cls] and [sep] here
    ).cuda()

    if rank==0:
        with open(log, "w") as f:
            pass

    # for getting training data
    sql = """
        SELECT g.id, g.seq_nc, g.length_nc, s.species_id
        FROM (SELECT id, seq_nc, length_nc, strain_id FROM gene WHERE id > %s AND strain_id IN (8654, 22145) ORDER BY id FETCH FIRST %s ROWS ONLY) AS g
        INNER JOIN strain AS s ON g.strain_id = s.id;
    """
    count = 10732
    gene_batch_size = 5500

    # for getting cross-species data
    translation = torch.load("translation_train.pt")

    # forward pass
    for i in range(epoch):
        # gene batch loader
        gbl = GeneBatchLoader(sql, gene_batch_size, count)

        for j, rows in enumerate(gbl):
            X = []
            label = []
            for last_id, seq, length, species_id in rows:
                if length//3 <= max_length:
                    seq = sanitize(seq)
                    seq = re.split('(...)',seq)[1:-1:2]
                    seq = list(map(lambda x: vocab.index(x), seq))
                    seq = torch.tensor(seq)
                    X.append(seq)
                    label.append(species_id)
            gbl.last_id = last_id
            
            X = pad_sequence(X)
            label = torch.tensor(label).unsqueeze(1)
            if X.size(0) < max_length:
                X = torch.cat((X, torch.zeros(max_length-X.size(0), X.size(1)).long()), dim=0)
            
            # for mlm task
            dataset = TensorDataset(X.T, label)

            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank
            )
            loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=sampler,
            )

            # for xlm task
            input = translation["input"]
            target = translation["target"]
            input_sp = translation["input_sp"]
            target_sp = translation["target_sp"]
            repeat = (X.size(1) // input.size(0)) + 1
            input = input.repeat(repeat, 1)
            target = target.repeat(repeat, 1)
            input_sp = input_sp.repeat(repeat, 1)
            target_sp = target_sp.repeat(repeat, 1)

            dataset = TensorDataset(input, target, input_sp, target_sp)

            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank
            )
            loader_trans = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=sampler,
            )

            for (x, label), (input, target, input_sp, target_sp) in zip(loader, loader_trans):
                # move to gpu
                x = x.cuda(non_blocking=True)
                label = label.expand_as(x).cuda(non_blocking=True)
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                input_sp = input_sp.cuda(non_blocking=True)
                target_sp = target_sp.cuda(non_blocking=True)

                optimizer.zero_grad()
                loss_mlm, loss_xlm = trainer(x, label, input, target, input_sp, target_sp)
                loss = loss_mlm + loss_xlm

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()

            if rank == 0:
                with open(log, "a") as f:
                    f.write("RANK({:0=2}) EPOCH({:0=3}%) STEP({:0=3}%) LOSS: MLM={:.4f} XLM={:.4f}\n".format(rank, int((i+1)*100/epoch), int((j+1)*100/len(gbl)), loss_mlm.item(), loss_xlm.item()))

        if rank==0:
            checkpoint = {
                'model': ddp_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)

    dist.destroy_process_group()