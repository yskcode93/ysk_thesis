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

from .file import GeneBatchLoader, sanitize, vocab

from .database import fetchall

def train(gpu, world_size, PretrainClass, config, n_epochs, batch_size, lr, warmup, use_apex,\
    strain_ids, log_interval, save_interval, log="./pretrain.log", checkpoint_path="./checkpoint.pt"):

    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    # define model
    model = PretrainClass(**config).to(gpu)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # construct DDP model
    ddp_model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu])
    # define loss function
    trainer = MLM(
        ddp_model,
        mask_token_id = vocab.index("<msk>"),          # the token id reserved for masking
        pad_token_id = vocab.index("<pad>"),           # the token id for padding
        mask_prob = 0.15,           # masking probability for masked language modeling
        replace_prob = 0.80,        # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
        num_tokens = 65,
        random_token_prob = 0.0,
        mask_ignore_token_ids = []  # other tokens to exclude from masking, include the [cls] and [sep] here
    ).cuda()

    if rank==0:
        with open(log, "w") as f:
            pass

    sql = """
        SELECT strain_id, seq_nc, length_nc FROM gene
        WHERE strain_id IN ({})
    """.format(",".join(["%s"]*len(strain_ids)))

    rows = fetchall(sql, strain_ids)
    X = []
    label = []
    for sid, seq, length in rows:
        if length//3 <= 700:
            seq = sanitize(seq)
            seq = re.split('(...)',seq)[1:-1:2]
            seq = list(map(lambda x: vocab.index(x), seq))
            seq = torch.tensor(seq)
            X.append(seq)
            label.append(strain_ids.index(sid))

    X = pad_sequence(X, batch_first=True)
    label = torch.tensor(label).unsqueeze(1)

    if X.size(1) < 700:
        X = torch.cat((X, torch.zeros(X.size(0), 700-X.size(1)).long()), dim=1)

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
        total_steps=n_epochs*len(loader),
        pct_start=warmup,
        anneal_strategy='linear'
    )

    for i in range(1, n_epochs+1):
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
            loss = trainer(x, sp=label)

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

        if rank == 0 and i % log_interval == 0:
            with open(log, "a") as f:
                f.write("EPOCH({:0=3}%) LOSS: MLM={:.4f} lr={:.4f}(1e-4) ELAPSED TIME: {:3.1f}s\n".format(int((i+1)*100/n_epochs), loss_avg, scheduler.get_last_lr()[0]*1e4, end-start))

        if rank==0 and i % save_interval == 0:
            checkpoint = {
                'model': ddp_model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'amp': amp.state_dict()
            } if use_apex else {
                'model': ddp_model.module.state_dict()
            }

            torch.save(checkpoint, checkpoint_path)

    dist.destroy_process_group()

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