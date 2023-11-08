# pytorch
import torch
import torch.nn as nn
import torch.optim as opt
import torch.optim.lr_scheduler as lrs
import torch.utils.data as dat
import torch.nn.utils as utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
# distributed learning
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# a pytorch extension fp-16
from apex.parallel import DistributedDataParallel as DDP_APEX
from apex import amp
# gene batch loader
from file import GeneBatchLoader, sanitize, vocab, write_fna_faa, write_fna_faa_2
# fetch all
from database import fetchall
from preprocess import fasta_to_data, fasta_to_data_aa, fasta_to_data_eos, right_shift_zeros

from fast_transformers.masking import LengthMask, TriangularCausalMask
from new_loss import cls_CrossEntropyLoss

# train test split
from sklearn.model_selection import KFold, StratifiedKFold

# base package
import re
import time
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import numpy as np
import glob
import re
import os

from torch.utils.data import DataLoader, TensorDataset, random_split

import math
from functools import reduce

from mlm_pytorch import MLM

# inference mode
MODE_GREEDY = 0
MODE_BEAM = 1


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from apex import amp
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import torch.optim as opt
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP_APEX
import torch.utils.data as dat
import torch.utils.data as utils
from torch.nn.utils import clip_grad_norm_ 
import os

def final_saksub_convert_test(ConverterClass, config, strain_ids, sp, n_epochs, n_folds, mode, device,\
    pretrain, PretrainClass, config_pretrain, pretrain_path_en, pretrain_path_de,\
    output_dir="./Result", batch_size=batch_size):
    
    thupath = "./Data/sak_to_sub/src_train.npy"
    X = np.load(thupath)

    X = torch.from_numpy(X.astype(np.int64)).clone()
    #通常データ
    if X.size(1) < 700:
        X = torch.cat((X, torch.zeros(X.size(0), 700-X.size(1)).long()), dim=1)

    X = X.to(device)
    
    # converter class
    pt = torch.load("{}/weight/weight_M2_8_8/sak_sub_3_{}.pt".format(output_dir, n_epochs), map_location=lambda storage, loc: storage.cuda(device))
    model = ConverterClass(**config).to(device)
    model.load_state_dict(pt["model"],strict=False)
    model.eval()
    del pt

    with torch.no_grad():
        for x in zip(torch.split(X, batch_size)):
                out = model(x_e, x, y_sp)
                out = torch.transpose(out, 1, 2)
                out = torch.argmax(out, dim=2)

                if out.size(1) < 700:
                    out = torch.cat((out.cpu(), torch.zeros(out.size(0), 700-out.size(1)).long()), dim=1)     
                outs.append(out.cpu())           
            out = torch.cat(outs)
            out = out

    write_fna_faa_2(out.cpu(), "{}/finetune/gen_for_8_8/gen_train_sak_sub_3".format(output_dir), "GEN")  

def final_saksub_convert(gpu, world_size, ConverterClass, config, weight_path, n_epochs, batch_size, lr, warmup, use_apex, finetune,\
        strain_ids, run_ids, direction, mode,\
        log_interval=1, save_interval=1, output_dir="./Result", random_state=1444\
    ):
    
    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    #train用
    train_src_tagpath = "./Final_Data/sak_to_sub/npy/src_train_tag.npy" 
    tag_src_train = np.load(train_src_tagpath)
    
    train_tgt_tagpath = "./Final_Data/sak_to_sub/npy/tgt_train_tag.npy" 
    tag_tgt_train = np.load(train_tgt_tagpath)
    
    train_src_path = "./Final_Data/sak_to_sub/npy/src_train.npy" 
    X_train = np.load(train_srcpath)
    
    train_tgt_path = "./Final_Data/sak_to_sub/npy/tgt_train.npy"
    Y_train = np.load(train_tgtpath)
    
    #変換タグ作成
    cv_tag_train = np.full((X_train.shape[0], 1), 119, dtype=np.float64)

    X_train = np.concatenate([tag_src_train, X_train, cv_tag_train, tag_tgt_train, Y_train], 1)
    X_train = right_shift_zeros(X_train)
    
    X_train = torch.from_numpy(X_train.astype(np.int64)).clone()
    Y_train = torch.from_numpy(Y_train.astype(np.int64)).clone()

    #valid用
    valid_src_tagpath = "./Final_Data/sak_to_sub/npy/src_valid_tag.npy" 
    tag_src_valid = np.load(valid_src_tagpath)
    
    valid_tgt_tagpath = "./Final_Data/sak_to_sub/npy/tgt_valid_tag.npy" 
    tag_tgt_valid = np.load(valid_tgt_tagpath)
    
    valid_src_path = "./Final_Data/sak_to_sub/npy/src_valid.npy" 
    X_valid = np.load(valid_srcpath)
    
    valid_tgt_path = "./Final_Data/sak_to_sub/npy/tgt_valid.npy"
    Y_valid = np.load(valid_tgtpath)
    
    #変換タグ作成
    cv_tag_valid = np.full((X_valid.shape[0], 1), 119, dtype=np.float64)

    X_valid = np.concatenate([tag_src_valid, X_valid, cv_tag_valid, tag_tgt_valid, Y_valid], 1)
    X_valid = right_shift_zeros(X_valid)
    
    X_valid = torch.from_numpy(X_valid.astype(np.int64)).clone()
    Y_valid = torch.from_numpy(Y_valid.astype(np.int64)).clone()
    
    max_length = 1405
    #max_length = 700
    
    if X_train.size(1) < max_length:
        X_train = torch.cat((X_train, torch.zeros(X_train.size(0), max_length-X_train.size(1)).long()), dim=1)

    if Y_train.size(1) < max_length:
        Y_train = torch.cat((Y_train, torch.zeros(Y_train.size(0), max_length-Y_train.size(1)).long()), dim=1)
        
    if X_valid.size(1) < max_length:
        X_valid = torch.cat((X_valid, torch.zeros(X_valid.size(0), max_length-X_valid.size(1)).long()), dim=1)

    if Y_valid.size(1) < max_length:
        Y_valid = torch.cat((Y_valid, torch.zeros(Y_valid.size(0), max_length-Y_valid.size(1)).long()), dim=1)
    
    #この時点ではXはtorch.float32でYがtorch.int64
    train_dataset = dat.TensorDataset(X_train, Y_train)
    valid_dataset = dat.TensorDataset(X_valid, Y_valid)

    # load pretrained converter
    model = ConverterClass(**config).to(gpu)
    #model = model.to(gpu)
    if finetune:
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(weight_path, map_location=map_location)
        model.load_state_dict(pt["model"])
        #model.freeze()
                
    optimizer = opt.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu], find_unused_parameters=True)
    # loader
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = dat.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)
    valid_loader = dat.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=valid_sampler)
    
    # scheduler
    scheduler = lrs.OneCycleLR(
        optimizer,
        lr,
        epochs = int(n_epochs),
        steps_per_epoch = len(train_loader),
        pct_start=warmup,
        anneal_strategy='linear'
    )

    transformer = nn.Transformer()
    mask = transformer.generate_square_subsequent_mask(sz=1405).to(gpu)
    
    for i in range(1, n_epochs+1):
        start = time.time()
        train_sampler.set_epoch(i)
        valid_sampler.set_epoch(i)
        model.train()  # 訓練モード
        total_train_loss = 0
            
        for x,y in train_loader:
            optimizer.zero_grad()
            x, y= x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            out = model(x, mask)
            loss = criterion(out.permute(0,2,1), y)
            
            loss.backward()
            # prevent gradient explosion
            clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
  
            total_train_loss += loss.item()
            
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()  # 評価モード
        total_valid_loss = 0
        with torch.no_grad():
            for x,y in valid_loader:
                x, y= x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                
                out = model(x, mask)
                loss = criterion(out.permute(0,2,1), y)
                
                total_valid_loss += loss.item()
            avg_valid_loss = total_valid_loss / len(valid_loader)
                

        if rank == 0 and i%log_interval==0:
            with open("{}/sak2sub_convert.log".format(output_dir), "a") as f:
                f.write("EPOCH({:0=3}%) TRAIN_LOSS: {:.4f} VALID_LOSS: {:.4f} lr={:.4f} x 1e-4 ELAPSED TIME: {:.4f}s\n".format(int(i*100/n_epochs), avg_train_loss, avg_valid_loss, scheduler.get_last_lr()[0]*1e4, time.time()-start))

        if rank == 0 and (i==0 or i%save_interval==0):
            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'scheduler': scheduler.state_dict()
            } if use_apex else {
                'model': model.module.state_dict(),
                'scheduler': scheduler.state_dict()
            }

            torch.save(checkpoint, "{}/weight/sak2sub/sak2sub_small_nc_{}.pt".format(output_dir, i))

def final_m2m_convert_valid(gpu, world_size, ConverterClass, config, weight_path, n_epochs, batch_size, lr, warmup, use_apex, finetune,\
        strain_ids, run_ids, direction, mode,\
        log_interval=1, save_interval=1, output_dir="./Result", random_state=1444\
    ):
    
    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    src_tagpath = "" 
    src_tag = np.load(src_tagpath)
    
    tgt_tagpath = "" 
    tgt_tag = np.load(tgt_tagpath)
    
    src_path = "" 
    X = np.load(src_path)
    
    tgt_path = ""
    Y = np.load(tgt_path)
    
    #変換タグ作成
    cv_tag = np.full((X.shape[0], 1), 119, dtype=np.float64)

    X = np.concatenate([src_tag, X, cv_tag, tgt_tag, Y], 1)
    X = right_shift_zeros(X)
    
    X = torch.from_numpy(X.astype(np.int64)).clone()
    Y = torch.from_numpy(Y.astype(np.int64)).clone()

    max_length = 1405
    #max_length = 700
    
    #通常データ
    if X.size(1) < max_length:
        X = torch.cat((X, torch.zeros(X.size(0), max_length-X.size(1)).long()), dim=1)
    if Y.size(1) < max_length:
        Y = torch.cat((Y, torch.zeros(Y.size(0), max_length-Y.size(1)).long()), dim=1)

    num_total = X.shape[0]
    num_train = int(num_total * 0.8)  # トレーニングデータの割合を80%とする
    num_valid = num_total - num_train
    
    #この時点ではXはtorch.float32でYがtorch.int64
    dataset = dat.TensorDataset(X, Y)
    #train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid], generator=torch.Generator().manual_seed(random_state))
    torch.manual_seed(random_state)

    # トレーニングデータセットとバリデーションデータセットに分割
    train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])
    
    # load pretrained converter
    model = ConverterClass(**config).to(gpu)
    #model = model.to(gpu)
    if finetune:
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(weight_path, map_location=map_location)
        model.load_state_dict(pt["model"])
        #model.freeze()
                
    optimizer = opt.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu], find_unused_parameters=True)
    # loader
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    #loader = dat.DataLoader(
    #    dataset=dataset,
    #    batch_size=batch_size,
    #    shuffle=False,
    #    num_workers=0,
    #    pin_memory=True,
    #    sampler=sampler
    #)
    train_loader = dat.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)
    valid_loader = dat.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=valid_sampler)
    
    # scheduler
    scheduler = lrs.OneCycleLR(
        optimizer,
        lr,
        epochs = int(n_epochs),
        steps_per_epoch = len(train_loader),
        pct_start=warmup,
        anneal_strategy='linear'
    )

    transformer = nn.Transformer()
    mask = transformer.generate_square_subsequent_mask(sz=1405).to(gpu)
    
    for i in range(1, n_epochs+1):
        start = time.time()
        train_sampler.set_epoch(i)
        valid_sampler.set_epoch(i)
        model.train()  # 訓練モード
        total_train_loss = 0
        correct_train = 0
        total_train = 0
            
        for x,y in train_loader:
            optimizer.zero_grad()
            x, y= x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            out = model(x, mask)
            loss = criterion(out.permute(0,2,1), y)
            
            _, predicted = torch.max(out, 2)
            mask_train = y != 0
            correct_train += predicted.eq(y).masked_select(mask_train).sum().item()
            total_train += mask_train.sum().item()
            
            loss.backward()
            # prevent gradient explosion
            clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
  
            total_train_loss += loss.item()
            
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        
        model.eval()  # 評価モード
        total_valid_loss = 0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for x,y in valid_loader:
                x, y= x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                
                out = model(x, mask)
                loss = criterion(out.permute(0,2,1), y)
                
                _, predicted = torch.max(out, 2)
                mask_valid = y != 0
                correct_valid += predicted.eq(y).masked_select(mask_valid).sum().item()
                total_valid += mask_valid.sum().item()
                
                total_valid_loss += loss.item()
            avg_valid_loss = total_valid_loss / len(valid_loader)
            valid_accuracy = correct_valid / total_valid
                

        if rank == 0 and i%log_interval==0:
            with open("{}/m2m_small.log".format(output_dir), "a") as f:
                f.write("EPOCH({:0=3}%) TRAIN_LOSS: {:.4f} TRAIN_ACC: {:.4f} VALID_LOSS: {:.4f} VALID_ACC: {:.4f} lr={:.4f} x 1e-4 ELAPSED TIME: {:.4f}s\n".format(int(i*100/n_epochs), avg_train_loss, train_accuracy, avg_valid_loss, valid_accuracy, scheduler.get_last_lr()[0]*1e4, time.time()-start))

        if rank == 0 and (i==0 or i%save_interval==0):
            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'scheduler': scheduler.state_dict()
            } if use_apex else {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }

            torch.save(checkpoint, "{}/weight/m2m/m2m_small_nc_{}.pt".format(output_dir, i))
                        
def final_pretrain_valid(gpu, world_size, ConverterClass, config, weight_path, n_epochs, batch_size, lr, warmup, use_apex, finetune,\
        strain_ids, run_ids, direction, mode,\
        pretrain, PretrainClass, config_pretrain, pretrain_path_en, pretrain_path_de,\
        log_interval=1, save_interval=1, output_dir="./Result", random_state=1444\
    ):
    
    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    train_tagpath = "Final_Data/50_select/datasets/stratified/pretrain_train_stratified_tag.npy" 
    train_tag = np.load(train_tagpath)
    
    #thupath = "Final_Data/50_select/52_all_nc.npy" 
    #X = np.load(thupath)
    
    train_subpath = "Final_Data/50_select/datasets/stratified/pretrain_train_stratified.npy"
    Y_train = np.load(train_subpath)

    X_train = np.concatenate([train_tag, Y_train], 1)
    
    X_train = torch.from_numpy(X_train.astype(np.int64)).clone()
    Y_train = torch.from_numpy(Y_train.astype(np.int64)).clone()
    
    valid_tagpath = "Final_Data/50_select/datasets/stratified/pretrain_valid_stratified_tag.npy" 
    valid_tag = np.load(valid_tagpath)
    
    #thupath = "Final_Data/50_select/52_all_nc.npy" 
    #X = np.load(thupath)
    
    valid_subpath = "Final_Data/50_select/datasets/stratified/pretrain_valid_stratified.npy"
    Y_valid = np.load(valid_subpath)

    X_valid = np.concatenate([valid_tag, Y_valid], 1)
    
    X_valid = torch.from_numpy(X_valid.astype(np.int64)).clone()
    Y_valid = torch.from_numpy(Y_valid.astype(np.int64)).clone()

    max_length = 703 if pretrain and "n_species" in config_pretrain else 702
    #max_length = 700
    
    #通常データ
    if X_train.size(1) < max_length:
        X_train = torch.cat((X_train, torch.zeros(X_train.size(0), max_length-X_train.size(1)).long()), dim=1)
    if Y_train.size(1) < max_length:
        Y_train = torch.cat((Y_train, torch.zeros(Y_train.size(0), max_length-Y_train.size(1)).long()), dim=1)
    if X_valid.size(1) < max_length:
        X_valid = torch.cat((X_valid, torch.zeros(X_valid.size(0), max_length-X_valid.size(1)).long()), dim=1)
    if Y_valid.size(1) < max_length:
        Y_valid = torch.cat((Y_valid, torch.zeros(Y_valid.size(0), max_length-Y_valid.size(1)).long()), dim=1)

    #torch.manual_seed(random_state)
    #torch.cuda.manual_seed_all(random_state)
    
    # ランダムなインデックスを取得
    #10000のスモールデータの部分
    #indices = torch.randperm(X.size(0))[:10000]
    #X = X[indices]
    #Y = Y[indices]
    
    #普段の全体の部分
    #num_total = X.shape[0]
    #num_train = int(num_total * 0.95)  # トレーニングデータの割合を95%とする
    #num_valid = num_total - num_train
    
    #この時点ではXはtorch.float32でYがtorch.int64
    #dataset = dat.TensorDataset(X, Y)

    # トレーニングデータセットとバリデーションデータセットに分割
    #train_dataset, valid_dataset = random_split(dataset, [num_train, num_valid])
    train_dataset = dat.TensorDataset(X_train, Y_train)
    valid_dataset = dat.TensorDataset(X_valid, Y_valid)
    
    # load pretrained converter
    model = ConverterClass(**config).to(gpu)
    #model = model.to(gpu)
    if finetune:
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(weight_path, map_location=map_location)
        model.load_state_dict(pt["model"])
        #model.freeze()
                
    optimizer = opt.AdamW(model.parameters())
        
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu], find_unused_parameters=True)
    # loader
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    valid_sampler = DistributedSampler(
        valid_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    #loader = dat.DataLoader(
    #    dataset=dataset,
    #    batch_size=batch_size,
    #    shuffle=False,
    #    num_workers=0,
    #    pin_memory=True,
    #    sampler=sampler
    #)
    train_loader = dat.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=train_sampler)
    valid_loader = dat.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, sampler=valid_sampler)
    
    # scheduler
    scheduler = lrs.OneCycleLR(
        optimizer,
        lr,
        epochs = int(n_epochs),
        steps_per_epoch = len(train_loader),
        pct_start=warmup,
        anneal_strategy='linear'
    )

    transformer = nn.Transformer()
    mask = transformer.generate_square_subsequent_mask(sz=702).to(gpu)
    
    for i in range(1, n_epochs+1):
        start = time.time()
        train_sampler.set_epoch(i)
        valid_sampler.set_epoch(i)
        model.train()  # 訓練モード
        total_train_loss = 0
        correct_train = 0
        total_train = 0
            
        for x,y in train_loader:
            optimizer.zero_grad()
            x, y= x.cuda(non_blocking=True), y.cuda(non_blocking=True)

            out = model(x, mask)
            loss = criterion(out.permute(0,2,1), y)
            
            _, predicted = torch.max(out, 2)
            mask_train = y != 0
            correct_train += predicted.eq(y).masked_select(mask_train).sum().item()
            total_train += mask_train.sum().item()
            
            loss.backward()
            # prevent gradient explosion
            clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
  
            total_train_loss += loss.item()
            
            scheduler.step()
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        
        model.eval()  # 評価モード
        total_valid_loss = 0
        correct_valid = 0
        total_valid = 0
        with torch.no_grad():
            for x,y in valid_loader:
                x, y= x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                
                out = model(x, mask)
                loss = criterion(out.permute(0,2,1), y)
                
                _, predicted = torch.max(out, 2)
                mask_valid = y != 0
                correct_valid += predicted.eq(y).masked_select(mask_valid).sum().item()
                total_valid += mask_valid.sum().item()
                
                total_valid_loss += loss.item()
            avg_valid_loss = total_valid_loss / len(valid_loader)
            valid_accuracy = correct_valid / total_valid
                

        if rank == 0 and i%log_interval==0:
            with open("{}/52_pretrain_stratified_1024_norm.log".format(output_dir), "a") as f:
                f.write("EPOCH({:0=3}%) TRAIN_LOSS: {:.4f} TRAIN_ACC: {:.4f} VALID_LOSS: {:.4f} VALID_ACC: {:.4f} lr={:.4f} x 1e-4 ELAPSED TIME: {:.4f}s\n".format(int(i*100/n_epochs), avg_train_loss, train_accuracy, avg_valid_loss, valid_accuracy, scheduler.get_last_lr()[0]*1e4, time.time()-start))

        if rank == 0 and (i==0 or i%save_interval==0):
            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'scheduler': scheduler.state_dict()
            } if use_apex else {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }

            torch.save(checkpoint, "{}/weight/pretrain_52_stratified_1024_norm/pretrain_all_nc_{}.pt".format(output_dir, (i)))
                                        
def final_pretrain(gpu, world_size, ConverterClass, config, weight_path, n_epochs, batch_size, lr, warmup, use_apex, finetune,\
        strain_ids, run_ids, direction, mode,\
        pretrain, PretrainClass, config_pretrain, pretrain_path_en, pretrain_path_de,\
        log_interval=1, save_interval=1, output_dir="./Result", random_state=1999\
    ):
    
    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    #pretrain model encoder
    if pretrain:
        model_pretrain_en = PretrainClass(**config_pretrain).to(gpu)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(pretrain_path_en, map_location=map_location)
        model_pretrain_en.load_state_dict(pt["model"])
        model_pretrain_en.eval()

    # pretrain model decoder
    if pretrain:
        model_pretrain_de = PretrainClass(**config_pretrain).to(gpu)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(pretrain_path_de, map_location=map_location)
        model_pretrain_de.load_state_dict(pt["model"])
        model_pretrain_de.eval()
    
    #thupath = "./CDSdata/alignment/src_train_rev.npy"
    thupath = "./CDSdata/alignment/one_to_one/src_train_rev.npy"
    
    #52菌種事前学習のやつ↓
    #thupath = "Final_Data/50_select/52_all_nc.npy" 
    #thupath = "./Data/sak_to_sub/src_train.npy"
    
    #finetune用
    #X_path = "./CDSdata/12_aa/nc_data/97%regulated/src.npy"
    X = np.load(thupath)
    subpath = "./CDSdata/alignment/one_to_one/src_train_rev.npy"
    
    #52菌種事前学習のやつ↓
    #subpath = "Final_Data/50_select/52_all_nc.npy"
    #subpath = "./Data/sak_to_sub/tgt_train.npy"
    
    #finetune用
    #Y_path = "./CDSdata/12_aa/nc_data/97%regulated/tgt.npy"
    Y = np.load(subpath)
    
    Y_sp = torch.zeros(1887,1, dtype=torch.int64)
    
    #52菌種事前学習のやつ↓
    #Y_sp = torch.zeros(212118,1, dtype=torch.int64)
    #Y_sp = torch.zeros(35039,1, dtype=torch.int64)
    X = torch.from_numpy(X.astype(np.int64)).clone()
    Y = torch.from_numpy(Y.astype(np.int64)).clone()
    
    max_length = 701 if pretrain and "n_species" in config_pretrain else 700
    #max_length = 700
    
    #通常データ
    if X.size(1) < max_length:
        X = torch.cat((X, torch.zeros(X.size(0), max_length-X.size(1)).long()), dim=1)

    if Y.size(1) < max_length:
        Y = torch.cat((Y, torch.zeros(Y.size(0), max_length-Y.size(1)).long()), dim=1)

    #この時点ではXはtorch.float32でYがtorch.int64
    dataset = dat.TensorDataset(X, Y, Y_sp)
    
    # load pretrained converter
    model = ConverterClass(**config).to(gpu)
    #model = model.to(gpu)
    if finetune:
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(weight_path, map_location=map_location)
        model.load_state_dict(pt["model"])
        #model.freeze()
        cnt = 0
        for param in model.parameters():
            if(cnt<16):
                param.requires_grad = False
                cnt+=1
            else:
                cnt+=1
        
    #最後の1層だけの重み固定の解除
    #model.layers[28].requires_grad = True
    #model.layers[29].requires_grad = True
    #model.layers[30].requires_grad = True
    #model.layers[31].requires_grad = True

    optimizer = opt.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss()

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu], find_unused_parameters=True)
    # loader
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    loader = dat.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    # scheduler
    scheduler = lrs.OneCycleLR(
        optimizer,
        lr,
        epochs = int(n_epochs),
        steps_per_epoch = len(loader),
        pct_start=warmup,
        anneal_strategy='linear'
    )

    transformer = nn.Transformer()
    mask = transformer.generate_square_subsequent_mask(sz=700).to(gpu)
    
    for i in range(1, n_epochs+1):
        start = time.time()
        sampler.set_epoch(i)
        avg_loss = 0
            
        for x, y, y_sp in loader:
            optimizer.zero_grad()
            x, y, y_sp = x.cuda(non_blocking=True), y.cuda(non_blocking=True), y_sp.cuda(non_blocking=True)
            if pretrain:
                with torch.no_grad():
                    x_e = model_pretrain_en.get_output(x)
                x_e = torch.transpose(x_e, 1, 2)
                out = model(x_e, x[:,1:], y_sp, y[:,1:]) if pretrain and "n_species" in config_pretrain else model(x_e, x, y_sp, y)
                out = torch.transpose(out, 1, 2)
            else:
                x = torch.tensor(x, dtype=torch.float32)
                #out = model(x, x, y_sp, y)
                out = model(x, mask)
                #out = torch.transpose(out, 1, 2)

            loss = criterion(out.permute(0,2,1), y[:, 1:] if pretrain and "n_species" in config_pretrain else y)
            #loss.requires_grad = True
            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                #loss.backward(retain_graph=True)
                loss.backward()

            # prevent gradient explosion
            clip_grad_norm_(model.parameters(), 1)
            

            avg_loss += loss.item()
            optimizer.step()
            scheduler.step()

        avg_loss /= len(loader)

        if rank == 0 and i%log_interval==0:
            with open("{}/for_10_24.log".format(output_dir), "a") as f:
                f.write("EPOCH({:0=3}%) LOSS: {:.4f} lr={:.4f} x 1e-4 ELAPSED TIME: {:.4f}s\n".format(int(i*100/n_epochs), avg_loss,\
                    scheduler.get_last_lr()[0]*1e4, time.time()-start))

        if rank == 0 and (i==0 or i%save_interval==0):
            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'scheduler': scheduler.state_dict()
            } if use_apex else {
                'model': model.module.state_dict(),
                'scheduler': scheduler.state_dict()
            }

            torch.save(checkpoint, "{}/weight/weight_M2_10_11/thu_check_{}.pt".format(output_dir, i))
            
            

def cross_validate_flocal_cnn(gpu, world_size, ConverterClass, weight_path, n_epochs, batch_size, lr, warmup, use_apex, finetune,\
        strain_ids, run_ids, direction, mode,\
        pretrain, PretrainClass, config_pretrain, pretrain_path_en, pretrain_path_de,\
        log_interval=1, save_interval=1, output_dir="./Result", random_state=1999\
    ):
    
    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    # pretrain model encoder
    if pretrain:
        model_pretrain_en = PretrainClass(**config_pretrain).to(gpu)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(pretrain_path_en, map_location=map_location)
        model_pretrain_en.load_state_dict(pt["model"])
        model_pretrain_en.eval()

    # pretrain model decoder
    if pretrain:
        model_pretrain_de = PretrainClass(**config_pretrain).to(gpu)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(pretrain_path_de, map_location=map_location)
        model_pretrain_de.load_state_dict(pt["model"])
        model_pretrain_de.eval()
    
    thupath = "./Result/pretrain/protein_family_stratified/finetune/src_train_1.fna"
    thupath = glob.glob(thupath)
    subpath = "./Result/pretrain/protein_family_stratified/finetune/tgt_train_1.fna"
    subpath = glob.glob(subpath)
    Y_sp = torch.zeros(1450,1, dtype=torch.int64)
    
    X = fasta_to_data(thupath, slen= 700)
    Y = fasta_to_data(subpath, slen= 700)
    #X.to('cuda:%d' % rank)
    #Y.to('cuda:%d' % rank)    
    #Y_sp.to('cuda:%d' % rank) 
    
    max_length = 701 if pretrain and "n_species" in config_pretrain else 700

    if X.size(1) < max_length:
        X = torch.cat((X, torch.zeros(X.size(0), max_length-X.size(1)).long()), dim=1)

    if Y.size(1) < max_length:
        Y = torch.cat((Y, torch.zeros(Y.size(0), max_length-Y.size(1)).long()), dim=1)
            
    #この時点ではXもYもtorch.int64
    path_w = './test_w5.txt'
    with open(path_w, mode='a') as f:
        f.write("入力X\n")
        f.write(str(X.dtype))
        f.write("\n")
        f.write(str(X.shape))
        f.write("\n") 
        f.write("入力Y\n")
        f.write(str(Y.dtype))
        f.write("\n")
        f.write(str(Y.shape))

    dataset = dat.TensorDataset(X, Y, Y_sp)
    
    # load pretrained converter
    model = ConverterClass()
    model = model.to(gpu)
    if finetune:
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(weight_path, map_location=map_location)
        model.load_state_dict(pt["model"])
        model.freeze()

    optimizer = opt.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss()

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu], find_unused_parameters=True)
    # loader
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    loader = dat.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    # scheduler
    scheduler = lrs.OneCycleLR(
        optimizer,
        lr,
        epochs = n_epochs,
        steps_per_epoch = len(loader),
        pct_start=warmup,
        anneal_strategy='linear'
    )

    for i in range(1, n_epochs+1):
        start = time.time()
        sampler.set_epoch(i)
        avg_loss = 0
            
        for x, y, y_sp in loader:
            optimizer.zero_grad()
            x, y, y_sp = x.cuda(non_blocking=True), y.cuda(non_blocking=True), y_sp.cuda(non_blocking=True)
            
            path_w = './test_w3.txt'
            with open(path_w, mode='a') as f:
                f.write("y.shape\n")
                f.write(str(y.shape))
                f.write("n")   

            if pretrain:
                with torch.no_grad():
                    x_e = model_pretrain_en.get_output(x)
                x_e = torch.transpose(x_e, 1, 2)
                out = model(x_e, x[:,1:], y_sp, y[:,1:]) if pretrain and "n_species" in config_pretrain else model(x_e, x, y_sp, y)
                out = torch.transpose(out, 1, 2)
                
            else:
                out = model(x, x, y_sp, y)

            if pretrain:
                #with torch.no_grad():
                #subtilisのデコーダー
                out = model_pretrain_de.get_output_decoder(out)
            path_w = './test_w5.txt'
            with open(path_w, mode='a') as f:
                f.write("out.shape\n")
                f.write(str(out.shape))
                f.write("\n") 
                f.write("y.shape\n")
                f.write(str(y.shape))
                f.write("\n") 
            loss = criterion(out.permute(0,2,1), y[:, 1:] if pretrain and "n_species" in config_pretrain else y)
            #loss.requires_grad = True
            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # prevent gradient explosion
            utils.clip_grad_norm_(model.parameters(), 1)

            avg_loss += loss.item()
            optimizer.step()
            scheduler.step()

        avg_loss /= len(loader)

        if rank == 0 and i%log_interval==0:
            with open("{}/finetune_cnn.log".format(output_dir), "a") as f:
                f.write("EPOCH({:0=3}%) LOSS: {:.4f} lr={:.4f} x 1e-4 ELAPSED TIME: {:.4f}s\n".format(int(i*100/n_epochs), avg_loss,\
                    scheduler.get_last_lr()[0]*1e4, time.time()-start))

        if rank == 0 and (i==0 or i%save_interval==0):
            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'scheduler': scheduler.state_dict()
            } if use_apex else {
                'model': model.module.state_dict(),
                'scheduler': scheduler.state_dict()
            }

            torch.save(checkpoint, "{}/weight/checkpoint_cnn_512_24lay_rev_enno{}.pt".format(output_dir, i))

def cross_validate_flocal(gpu, world_size, ConverterClass, config, weight_path, n_epochs, batch_size, lr, warmup, use_apex, finetune,\
        strain_ids, run_ids, direction, mode,\
        pretrain, PretrainClass, config_pretrain, pretrain_path_en, pretrain_path_de,\
        log_interval=1, save_interval=1, output_dir="./Result", random_state=1999\
    ):
    
    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    # pretrain model encoder
    if pretrain:
        model_pretrain_en = PretrainClass(**config_pretrain).to(gpu)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(pretrain_path_en, map_location=map_location)
        model_pretrain_en.load_state_dict(pt["model"])
        model_pretrain_en.eval()

    # pretrain model decoder
    if pretrain:
        model_pretrain_de = PretrainClass(**config_pretrain).to(gpu)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(pretrain_path_de, map_location=map_location)
        model_pretrain_de.load_state_dict(pt["model"])
        model_pretrain_de.eval()
    
    thupath = "./Result/pretrain/protein_family_stratified/finetune/src_train_1.fna"
    thupath = glob.glob(thupath)
    subpath = "./Result/pretrain/protein_family_stratified/finetune/tgt_train_1.fna"
    subpath = glob.glob(subpath)
    Y_sp = torch.zeros(1450,1, dtype=torch.int64)
    
    X = fasta_to_data(thupath, slen= 700)
    Y = fasta_to_data(subpath, slen= 700)
    #X.to('cuda:%d' % rank)
    #Y.to('cuda:%d' % rank)    
    #Y_sp.to('cuda:%d' % rank) 
    
    max_length = 701 if pretrain and "n_species" in config_pretrain else 700

    if X.size(1) < max_length:
        X = torch.cat((X, torch.zeros(X.size(0), max_length-X.size(1)).long()), dim=1)

    if Y.size(1) < max_length:
        Y = torch.cat((Y, torch.zeros(Y.size(0), max_length-Y.size(1)).long()), dim=1)
            
    dataset = dat.TensorDataset(X, Y, Y_sp)
    
    # load pretrained converter
    model = ConverterClass(**config).to(gpu)
    if finetune:
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(weight_path, map_location=map_location)
        model.load_state_dict(pt["model"])
        model.freeze()

    optimizer = opt.AdamW(model.parameters())
    criterion = cls_CrossEntropyLoss()

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu], find_unused_parameters=True)
    # loader
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )
    loader = dat.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    # scheduler
    scheduler = lrs.OneCycleLR(
        optimizer,
        lr,
        epochs = n_epochs,
        steps_per_epoch = len(loader),
        pct_start=warmup,
        anneal_strategy='linear'
    )

    for i in range(1, n_epochs+1):
        start = time.time()
        sampler.set_epoch(i)
        avg_loss = 0
            
        for x, y, y_sp in loader:
            optimizer.zero_grad()
            x, y, y_sp = x.cuda(non_blocking=True), y.cuda(non_blocking=True), y_sp.cuda(non_blocking=True)

            if pretrain:
                #with torch.no_grad():
                x_e = model_pretrain_en.get_output(x)

                out = model(x_e, x[:,1:], y_sp, y[:,1:]) if pretrain and "n_species" in config_pretrain else model(x_e, x, y_sp, y)
            else:
                out = model(x, x, y_sp, y)

            if pretrain:
                #with torch.no_grad():
                #subtilisのデコーダー
                out = model_pretrain_de.get_output_decoder(out)
                
            loss = criterion(out.permute(0,2,1), y[:, 1:] if pretrain and "n_species" in config_pretrain else y)
            #loss.requires_grad = True
            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # prevent gradient explosion
            utils.clip_grad_norm_(model.parameters(), 1)

            avg_loss += loss.item()
            optimizer.step()
            scheduler.step()

        avg_loss /= len(loader)

        if rank == 0 and i%log_interval==0:
            with open("{}/finetune_cluster.log".format(output_dir), "a") as f:
                f.write("EPOCH({:0=3}%) LOSS: {:.4f} lr={:.4f} x 1e-4 ELAPSED TIME: {:.4f}s\n".format(int(i*100/n_epochs), avg_loss,\
                    scheduler.get_last_lr()[0]*1e4, time.time()-start))

        if rank == 0 and (i==0 or i%save_interval==0):
            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'scheduler': scheduler.state_dict()
            } if use_apex else {
                'model': model.module.state_dict(),
                'scheduler': scheduler.state_dict()
            }

            torch.save(checkpoint, "{}/weight/checkpoint_cnn_{}.pt".format(output_dir, i))


def check(gpu, world_size, ConverterClass, config, weight_path, n_epochs, batch_size, lr, warmup, use_apex, finetune,\
        strain_ids, run_ids, direction, mode,\
        pretrain, PretrainClass, config_pretrain, pretrain_path_en, pretrain_path_de,\
        log_interval=1, save_interval=1, output_dir="./Result", random_state=1999\
    ):

    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    # 
    sql = """
        SELECT g1.seq_nc, g2.seq_nc, g1.strain_id, g2.strain_id, g1.id, g2.id FROM gene_gene as gg
        INNER JOIN gene AS g1 ON gg.gene_1_id=g1.id
        INNER JOIN gene AS g2 ON gg.gene_2_id=g2.id
        WHERE gg.run_id IN ({}) AND g1.length_nc <= 2100 AND g2.length_nc <= 2100 AND gg.length_ratio BETWEEN 0.97 AND 1.03;
    """.format(",".join(["%s"]*len(run_ids)))

    X, Y, X_sp, Y_sp, X_id, Y_id = [], [], [], [], [], []

    for x, y, x_sp, y_sp, x_id, y_id in fetchall(sql, run_ids):
        x, y = sanitize(x), sanitize(y)
        x, y = re.split('(...)',x)[1:-1:2], re.split('(...)',y)[1:-1:2]
        x, y = list(map(lambda x: vocab.index(x), x)), list(map(lambda x: vocab.index(x), y))
        if pretrain and "n_species" in config_pretrain:
            x, y = [len(vocab)] + x, [len(vocab)] + y
        x, y = torch.tensor(x), torch.tensor(y)
        x_sp, y_sp = 0, 1#strain_ids.index(x_sp), strain_ids.index(y_sp)

        if direction==0:
            X += [x, y]
            Y += [y, x]
            X_sp += [x_sp, y_sp]
            Y_sp += [y_sp, x_sp]
            X_id += [x_id, y_id]
            Y_id += [y_id, x_id]
        elif direction==1:
            X.append(x)
            Y.append(y)
            X_sp.append(x_sp)
            Y_sp.append(y_sp)
            X_id.append(x_id)
            Y_id.append(y_id)
        elif direction==2:
            X.append(y)
            Y.append(x)
            X_sp.append(y_sp)
            Y_sp.append(x_sp)
            X_id.append(y_id)
            Y_id.append(x_id)

    X, Y = pad_sequence(X, batch_first=True), pad_sequence(Y, batch_first=True)
    X_sp, Y_sp = torch.tensor(X_sp).unsqueeze(1), torch.tensor(Y_sp).unsqueeze(1)
    X_id, Y_id = np.array(X_id), np.array(Y_id)
 
    max_length = 701 if pretrain and "n_species" in config_pretrain else 700

    if X.size(1) < max_length:
        X = torch.cat((X, torch.zeros(X.size(0), max_length-X.size(1)).long()), dim=1)

    if Y.size(1) < max_length:
        Y = torch.cat((Y, torch.zeros(Y.size(0), max_length-Y.size(1)).long()), dim=1)
        
    path_w = './test_w.txt'
        
    with open(path_w, mode='a') as f:
        f.write(str(X))
        f.write("\n")
        f.write(str(X.shape))
        f.write("\n")        

    # pretrain model encoder
    if pretrain:
        model_pretrain_en = PretrainClass(**config_pretrain).to(gpu)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(pretrain_path_en, map_location=map_location)
        model_pretrain_en.load_state_dict(pt["model"])
        model_pretrain_en.eval()

    # pretrain model decoder
    if pretrain:
        model_pretrain_de = PretrainClass(**config_pretrain).to(gpu)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(pretrain_path_de, map_location=map_location)
        model_pretrain_de.load_state_dict(pt["model"])
        model_pretrain_de.eval()

    # cross validation
    kf = KFold(shuffle=True, random_state=random_state)
    Y_id_set = np.unique(Y_id)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(Y_id_set)):
        Y_id_set_test = Y_id_set[test_idx]
        test_idx = np.isin(Y_id, Y_id_set_test)
        train_idx = ~ test_idx
        
        if(fold_idx != 0):
            break
            
        # write only source and target
                
        dataset = dat.TensorDataset(X[train_idx], Y[train_idx], Y_sp[train_idx])
        
        model = ConverterClass(**config).to(gpu)
        if finetune:
            dist.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            pt = torch.load(weight_path, map_location=map_location)
            model.load_state_dict(pt["model"])
            model.freeze()
  
        optimizer = opt.AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss()

        if use_apex:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

        model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu], find_unused_parameters=True)
        # loader
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank
        )
        loader = dat.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=sampler
        )

        # scheduler
        scheduler = lrs.OneCycleLR(
            optimizer,
            lr,
            epochs = n_epochs,
            steps_per_epoch = len(loader),
            pct_start=warmup,
            anneal_strategy='linear'
        )

        for x, y, y_sp in loader:
            optimizer.zero_grad()
            x, y, y_sp = x.cuda(non_blocking=True), y.cuda(non_blocking=True), y_sp.cuda(non_blocking=True)
 
            if pretrain:

                x_e = model_pretrain_en.get_output(x)
                
                path_w = './test_w.txt'
                with open(path_w, mode='a') as f:
                    f.write("x_e\n")
                    f.write(str(x_e))
                    f.write("\n")
                    f.write(str(x_e.shape))

                with open(path_w, mode='a') as f:
                    f.write("x\n")
                    f.write(str(x))
                    f.write("\n")
                    f.write(str(x.shape))
                out = model(x_e, x[:,1:], y_sp, y[:,1:]) if pretrain and "n_species" in config_pretrain else model(x_e, x, y_sp, y)
                with open(path_w, mode='a') as f:
                    f.write("out\n")
                    f.write(str(out))
                    f.write("\n")
                    f.write(str(out.shape))

        
def cross_validate(gpu, world_size, ConverterClass, config, weight_path, n_epochs, batch_size, lr, warmup, use_apex, finetune,\
        strain_ids, run_ids, direction, mode,\
        pretrain, PretrainClass, config_pretrain, pretrain_path_en, pretrain_path_de,\
        log_interval=1, save_interval=1, output_dir="./Result", random_state=1999\
    ):
    """This function conducts k-fold cross validation of seq-to-seq transformer.

    Pass the function to torch.multiprocessing.spawn to start distributed learning on multiple gpus.

    Argument:
    gpu -- Integer: internal id for each gpu
    world_size -- Integer: total count of gpus
    ConverterClass -- Class: custom model class that extends torch.nn.Module
    config -- Dictionary: arguments of ConverterClass
    weight_path -- String: Path in which the model weights are saved
    n_epochs -- Integer: Epoch for training
    batch_size -- Integer: Batch size on each gpu
    lr -- Float: maximum learning rate for training
    warmup -- Float: ratio of warmup epochs compared to 'n_epochs'
    use_apex -- Boolean: whether or not A Pytorch Extension(apex) is used to get training much faster
    finetune -- Boolean: whether or not the model weights start from checkpoints provided by 'weight_path'
    strain_ids -- List(Integer): dummy argument
    run_ids -- List(Integer): list of run ids that are included to training dataset
    direction -- Integer{0,1,2}: direction for conversion 0(X->Y,Y->X), 1(X->Y), 2(Y->X)
    mode -- Integer: dummy argument
    pretrain -- Boolean: whether or not output feature vectors of the pretrained model are used in training.
    PretrainClass -- Class: custom model class that extends torch.nn.Module
    config_pretrain -- Dictionary: arguments of PretrainClass
    pretrain_path -- String: Path in which the model weights are saved
    log_interval -- Integer: training is logged once in 'log_interval' epochs
    save_interval -- Ineger: the model weights are saved to 'weight_path' once in 'save_interval'
    output_dir -- String: directory for saving results. This directory should have two child directories, ./finetune and ./weights.
    random_state -- Integer: seed value for spliting dataset.

    Return:
    None
    """

    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    # 
    sql = """
        SELECT g1.seq_nc, g2.seq_nc, g1.strain_id, g2.strain_id, g1.id, g2.id FROM gene_gene as gg
        INNER JOIN gene AS g1 ON gg.gene_1_id=g1.id
        INNER JOIN gene AS g2 ON gg.gene_2_id=g2.id
        WHERE gg.run_id IN ({}) AND g1.length_nc <= 2100 AND g2.length_nc <= 2100 AND gg.length_ratio BETWEEN 0.97 AND 1.03;
    """.format(",".join(["%s"]*len(run_ids)))

    X, Y, X_sp, Y_sp, X_id, Y_id = [], [], [], [], [], []

    for x, y, x_sp, y_sp, x_id, y_id in fetchall(sql, run_ids):
        x, y = sanitize(x), sanitize(y)
        x, y = re.split('(...)',x)[1:-1:2], re.split('(...)',y)[1:-1:2]
        x, y = list(map(lambda x: vocab.index(x), x)), list(map(lambda x: vocab.index(x), y))
        if pretrain and "n_species" in config_pretrain:
            x, y = [len(vocab)] + x, [len(vocab)] + y
        x, y = torch.tensor(x), torch.tensor(y)
        x_sp, y_sp = 0, 1#strain_ids.index(x_sp), strain_ids.index(y_sp)

        if direction==0:
            X += [x, y]
            Y += [y, x]
            X_sp += [x_sp, y_sp]
            Y_sp += [y_sp, x_sp]
            X_id += [x_id, y_id]
            Y_id += [y_id, x_id]
        elif direction==1:
            X.append(x)
            Y.append(y)
            X_sp.append(x_sp)
            Y_sp.append(y_sp)
            X_id.append(x_id)
            Y_id.append(y_id)
        elif direction==2:
            X.append(y)
            Y.append(x)
            X_sp.append(y_sp)
            Y_sp.append(x_sp)
            X_id.append(y_id)
            Y_id.append(x_id)

    X, Y = pad_sequence(X, batch_first=True), pad_sequence(Y, batch_first=True)
    X_sp, Y_sp = torch.tensor(X_sp).unsqueeze(1), torch.tensor(Y_sp).unsqueeze(1)
    X_id, Y_id = np.array(X_id), np.array(Y_id)

    max_length = 701 if pretrain and "n_species" in config_pretrain else 700

    if X.size(1) < max_length:
        X = torch.cat((X, torch.zeros(X.size(0), max_length-X.size(1)).long()), dim=1)

    if Y.size(1) < max_length:
        Y = torch.cat((Y, torch.zeros(Y.size(0), max_length-Y.size(1)).long()), dim=1)

    # pretrain model encoder
    if pretrain:
        model_pretrain_en = PretrainClass(**config_pretrain).to(gpu)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(pretrain_path_en, map_location=map_location)
        model_pretrain_en.load_state_dict(pt["model"])
        model_pretrain_en.eval()

    # pretrain model decoder
    if pretrain:
        model_pretrain_de = PretrainClass(**config_pretrain).to(gpu)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(pretrain_path_de, map_location=map_location)
        model_pretrain_de.load_state_dict(pt["model"])
        model_pretrain_de.eval()

    # cross validation
    kf = KFold(shuffle=True, random_state=random_state)
    Y_id_set = np.unique(Y_id)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(Y_id_set)):
        Y_id_set_test = Y_id_set[test_idx]
        test_idx = np.isin(Y_id, Y_id_set_test)
        train_idx = ~ test_idx
        
        if(fold_idx != 0):
            break
            
        # write only source and target
        
        if rank == 0:
            write_fna_faa(X[test_idx, 1:] if pretrain and "n_species" in config_pretrain else X[test_idx], "{}/finetune/src_{}".format(output_dir, fold_idx+1), "SRC")
            write_fna_faa(Y[test_idx, 1:] if pretrain and "n_species" in config_pretrain else Y[test_idx], "{}/finetune/tgt_{}".format(output_dir, fold_idx+1), "TGT")
            write_fna_faa(X[train_idx, 1:] if pretrain and "n_species" in config_pretrain else X[train_idx], "{}/finetune/src_train_{}".format(output_dir, fold_idx+1), "SRC")
            write_fna_faa(Y[train_idx, 1:] if pretrain and "n_species" in config_pretrain else Y[train_idx], "{}/finetune/tgt_train_{}".format(output_dir, fold_idx+1), "TGT")
            
        dataset = dat.TensorDataset(X[train_idx], Y[train_idx], Y_sp[train_idx])

        #テストの変換元と変換先
        #write_fna_faa(X[test_idx, 1:] if pretrain and "n_species" in config_pretrain else X[test_idx], "{}/finetune/src_{}".format(output_dir, fold_idx+1), "SRC")
        #write_fna_faa(Y[test_idx, 1:] if pretrain and "n_species" in config_pretrain else Y[test_idx], "{}/finetune/tgt_{}".format(output_dir, fold_idx+1), "TGT")

        #dataset = dat.TensorDataset(X[train_idx], Y[train_idx], Y_sp[train_idx])
        #write_fna_faa(X[train_idx, 1:] if pretrain and "n_species" in config_pretrain else X[train_idx], "{}/finetune/src_train_{}".format(output_dir, fold_idx+1), "SRC")
        #write_fna_faa(Y[train_idx, 1:] if pretrain and "n_species" in config_pretrain else Y[train_idx], "{}/finetune/tgt_train_{}".format(output_dir, fold_idx+1), "TGT")
        
        
        # load pretrained converter
        model = ConverterClass(**config).to(gpu)
        if finetune:
            dist.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            pt = torch.load(weight_path, map_location=map_location)
            model.load_state_dict(pt["model"])
            model.freeze()

        optimizer = opt.AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss()

        if use_apex:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

        model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu], find_unused_parameters=True)
        # loader
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank
        )
        loader = dat.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=sampler
        )

        # scheduler
        scheduler = lrs.OneCycleLR(
            optimizer,
            lr,
            epochs = n_epochs,
            steps_per_epoch = len(loader),
            pct_start=warmup,
            anneal_strategy='linear'
        )

        for i in range(1, n_epochs+1):
            start = time.time()
            sampler.set_epoch(i)
            avg_loss = 0
            
            for x, y, y_sp in loader:
                optimizer.zero_grad()
                x, y, y_sp = x.cuda(non_blocking=True), y.cuda(non_blocking=True), y_sp.cuda(non_blocking=True)

                if pretrain:
                    #with torch.no_grad():
                    x_e = model_pretrain_en.get_output(x)

                    out = model(x_e, x[:,1:], y_sp, y[:,1:]) if pretrain and "n_species" in config_pretrain else model(x_e, x, y_sp, y)
                else:
                    out = model(x, x, y_sp, y)

                if pretrain:
                    #with torch.no_grad():
                    #subtilisのデコーダー
                    out = model_pretrain_de.get_output_decoder_2(out)
                
                loss = criterion(out.permute(0,2,1), y[:, 1:] if pretrain and "n_species" in config_pretrain else y)
                #loss.requires_grad = True
                if use_apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # prevent gradient explosion
                utils.clip_grad_norm_(model.parameters(), 1)

                avg_loss += loss.item()
                optimizer.step()
                scheduler.step()

            avg_loss /= len(loader)

            if rank == 0 and i%log_interval==0:
                with open("{}/finetune.log".format(output_dir), "a") as f:
                    f.write("FOLD {} EPOCH({:0=3}%) LOSS: {:.4f} lr={:.4f} x 1e-4 ELAPSED TIME: {:.4f}s\n".format(fold_idx+1, int(i*100/n_epochs), avg_loss,\
                        scheduler.get_last_lr()[0]*1e4, time.time()-start))

            if rank == 0 and (i==0 or i%save_interval==0):
                checkpoint = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                    'scheduler': scheduler.state_dict()
                } if use_apex else {
                    'model': model.module.state_dict(),
                    'scheduler': scheduler.state_dict()
                }

                torch.save(checkpoint, "{}/weight/checkpoint_{}_{}.pt".format(output_dir, fold_idx+1, i))


def test_from_fasta_flocal(ConverterClass, config, strain_ids, sp, n_epochs, n_folds, mode, device,\
    pretrain, PretrainClass, config_pretrain, pretrain_path_en, pretrain_path_de,\
    output_dir="./Result", batch_size=100):

    #for i in tqdm(range(1, n_folds+1)):
    # source sequences
    with open("{}/finetune/src_1.fna".format(output_dir), "r") as f:
        X = list(SeqIO.parse(f, "fasta"))

    X = [sanitize(str(x.seq)) for x in X]
    X = [re.split('(...)',x)[1:-1:2] for x in X]
    X = [torch.tensor(list(map(lambda x: vocab.index(x), x))) for x in X]
    X = pad_sequence(X, batch_first=True)
    if X.size(1) < 700:
        X = torch.cat((X, torch.zeros(X.size(0), 700-X.size(1)).long()), dim=1)

    Y_sp = strain_ids.index(sp)

    Y_sp = torch.tensor(Y_sp).unsqueeze(0).repeat(X.size(0), 1)

    X, Y_sp = X.to(device), Y_sp.to(device)
    # converter class
    pt = torch.load("{}/weight/checkpoint_cnn_512_24lay_rev_enno{}.pt".format(output_dir, n_epochs), map_location=lambda storage, loc: storage.cuda(device))
    #model = ConverterClass(**config).to(device)
    model = ConverterClass()
    model = model.to(device)
    #print(model)
    model.load_state_dict(pt["model"],strict=False)
    model.eval()
    del pt

    # pretrain class encoder
    if pretrain:
        pt = torch.load(pretrain_path_en, map_location=lambda storage, loc: storage.cuda(device))
        model_pretrain_en = PretrainClass(**config_pretrain).to(device)
        model_pretrain_en.load_state_dict(pt["model"])
        model_pretrain_en.eval()
        del pt
            
    # pretrain class decoder
    if pretrain:
        pt = torch.load(pretrain_path_de, map_location=lambda storage, loc: storage.cuda(device))
        model_pretrain_de = PretrainClass(**config_pretrain).to(device)
        model_pretrain_de.load_state_dict(pt["model"])
        model_pretrain_de.eval()
        del pt

    with torch.no_grad():
        if pretrain:
            cls = torch.tensor([[len(vocab)]]*X.size(0)).to(device)
            X_e = model_pretrain_en.get_output(torch.cat((cls, X), dim=1)) if "n_species" in config_pretrain else model_pretrain_en.get_output(X)
            X_e = torch.transpose(X_e, 1, 2)
        else:
            X_e = X

        if mode == MODE_GREEDY:
            out = model.infer_greedily(X_e, X, Y_sp, X.size(1), [17, 19, 25])
        elif mode == MODE_BEAM:
            outs = []
            for x_e, x, y_sp in zip(torch.split(X_e, batch_size), torch.split(X, batch_size), torch.split(Y_sp, batch_size)):
                out = model(x_e, x, y_sp)
                out = torch.transpose(out, 1, 2)
                if pretrain:
                    out = model_pretrain_de.get_output_decoder(out)
                    
                print("out")
                print(out.shape)
                print(out)
                    
                out = torch.argmax(out, dim=2)

                print("arg後out")
                print(out.shape)
                print(out)
                    
                if out.size(1) < 700:
                    out = torch.cat((out.cpu(), torch.zeros(out.size(0), 700-out.size(1)).long()), dim=1)
                   
                outs.append(out.cpu())         
                
            out = torch.cat(outs)
            out = out
                
                
            #out = model_pretrain_de.get_output_decoder(torch.cat((cls, out), dim=1)) if "n_species" in config_pretrain else model_pretrain_de.get_output_decoder(out)       
    #print("出力")
    print(out)
    print(out.shape)

    write_fna_faa(out.cpu(), "{}/finetune/gen_cnn_512_24lay_rev_enno".format(output_dir), "GEN")   

def test_from_train_flocal(ConverterClass, config, strain_ids, sp, n_epochs, n_folds, mode, device,\
    pretrain, PretrainClass, config_pretrain, pretrain_path_en, pretrain_path_de,\
    output_dir="./Result", batch_size=100):

    #for i in tqdm(range(1, n_folds+1)):
    # source sequences
    with open("{}/finetune/src_train_1.fna".format(output_dir), "r") as f:
        X = list(SeqIO.parse(f, "fasta"))

    X = [sanitize(str(x.seq)) for x in X]
    X = [re.split('(...)',x)[1:-1:2] for x in X]
    X = [torch.tensor(list(map(lambda x: vocab.index(x), x))) for x in X]
    X = pad_sequence(X, batch_first=True)
    if X.size(1) < 700:
        X = torch.cat((X, torch.zeros(X.size(0), 700-X.size(1)).long()), dim=1)

    Y_sp = strain_ids.index(sp)

    Y_sp = torch.tensor(Y_sp).unsqueeze(0).repeat(X.size(0), 1)

    X, Y_sp = X.to(device), Y_sp.to(device)
    # converter class
    pt = torch.load("{}/weight/checkpoint_cnn_512_24lay_rev_enno{}.pt".format(output_dir, n_epochs), map_location=lambda storage, loc: storage.cuda(device))
    model = ConverterClass()
    model = model.to(device)
    #print(model)
    model.load_state_dict(pt["model"],strict=False)
    model.eval()
    del pt

    # pretrain class encoder
    if pretrain:
        pt = torch.load(pretrain_path_en, map_location=lambda storage, loc: storage.cuda(device))
        model_pretrain_en = PretrainClass(**config_pretrain).to(device)
        model_pretrain_en.load_state_dict(pt["model"])
        model_pretrain_en.eval()
        del pt
            
    # pretrain class decoder
    if pretrain:
        pt = torch.load(pretrain_path_de, map_location=lambda storage, loc: storage.cuda(device))
        model_pretrain_de = PretrainClass(**config_pretrain).to(device)
        model_pretrain_de.load_state_dict(pt["model"])
        model_pretrain_de.eval()
        del pt

    with torch.no_grad():
        if pretrain:
            cls = torch.tensor([[len(vocab)]]*X.size(0)).to(device)
            X_e = model_pretrain_en.get_output(torch.cat((cls, X), dim=1)) if "n_species" in config_pretrain else model_pretrain_en.get_output(X)
            X_e = torch.transpose(X_e, 1, 2)
        else:
            X_e = X

        if mode == MODE_GREEDY:
            out = model.infer_greedily(X_e, X, Y_sp, X.size(1), [17, 19, 25])
        elif mode == MODE_BEAM:
            outs = []
            for x_e, x, y_sp in zip(torch.split(X_e, batch_size), torch.split(X, batch_size), torch.split(Y_sp, batch_size)):
                out = model(x_e, x, y_sp)
                out = torch.transpose(out, 1, 2)
                if pretrain:
                    out = model_pretrain_de.get_output_decoder(out)
                    
                #lfc = nn.Linear(67, 66).to(device)
                #out = out.cuda()
                #out = lfc(out)
                    
                print("out")
                print(out.shape)
                print(out)
                    
                out = torch.argmax(out, dim=2)
                #log_lik, out = torch.topk(F.log_softmax(out[:,-1], dim=1), 5)
                #log_lik, out = torch.topk(F.log_softmax(out, dim=1), 5)
                #out = out.view(-1, 1)
                    
                if out.size(1) < 700:
                    out = torch.cat((out.cpu(), torch.zeros(out.size(0), 700-out.size(1)).long()), dim=1)
                   
                outs.append(out.cpu())         
                
            out = torch.cat(outs)
            out = out
                
                
            #out = model_pretrain_de.get_output_decoder(torch.cat((cls, out), dim=1)) if "n_species" in config_pretrain else model_pretrain_de.get_output_decoder(out)  
            
    #print("出力")
    print(out)
    #print(out.shape)
    write_fna_faa(out.cpu(), "{}/finetune/gen_train_cnn_512_24lay_rev_enno".format(output_dir), "GEN")   

def test_from_npy_flocal(ConverterClass, config, strain_ids, sp, n_epochs, n_folds, mode, device,\
    pretrain, PretrainClass, config_pretrain, pretrain_path_en, pretrain_path_de,\
    output_dir="./Result", batch_size=100):

    #for i in tqdm(range(1, n_folds+1)):
    # source sequences
    #with open("{}/finetune/src_1.fna".format(output_dir), "r") as f:
    #    X = list(SeqIO.parse(f, "fasta"))

    #X = [sanitize(str(x.seq)) for x in X]
    #X = [re.split('(...)',x)[1:-1:2] for x in X]
    #X = [torch.tensor(list(map(lambda x: vocab.index(x), x))) for x in X]
    #X = pad_sequence(X, batch_first=True)
    #if X.size(1) < 700:
    #    X = torch.cat((X, torch.zeros(X.size(0), 700-X.size(1)).long()), dim=1)

    vocab_size = 700
    emb_dim = 512
    embeddings = nn.Embedding(vocab_size, emb_dim)
    
    #thupath = "./CDSdata/12_aa/nc_data/97%regulated/src.npy"
    #thupath = "./CDSdata/alignment/one_to_one/src_train_rev.npy"
    thupath = "./Data/sak_to_sub/src_train.npy"
    X = np.load(thupath)
    #subpath = "./CDSdata/alignment/tgt_test_rev.npy"
    #Y = np.load(subpath)
    Y_sp = torch.zeros(804,1, dtype=torch.int64)
    X = torch.from_numpy(X.astype(np.int64)).clone()
    #Y = torch.from_numpy(Y.astype(np.int64)).clone()
    #通常データ
    if X.size(1) < 700:
        X = torch.cat((X, torch.zeros(X.size(0), 700-X.size(1)).long()), dim=1)
    #onehot化
    #if X.size(2) < 700:
    #    X = torch.cat((X, torch.zeros(X.size(1), 700-X.size(2)).long()), dim=1)    
    
    Y_sp = strain_ids.index(sp)

    Y_sp = torch.tensor(Y_sp).unsqueeze(0).repeat(X.size(0), 1)

   

    X, Y_sp = X.to(device), Y_sp.to(device)
    # converter class
    pt = torch.load("{}/weight/weight_M2_8_8/sak_sub_3_{}.pt".format(output_dir, n_epochs), map_location=lambda storage, loc: storage.cuda(device))
    #model = ConverterClass(**config).to(device)
    model = ConverterClass()
    model = model.to(device)
    #print(model)
    model.load_state_dict(pt["model"],strict=False)
    model.eval()
    del pt

    # pretrain class encoder
    if pretrain:
        pt = torch.load(pretrain_path_en, map_location=lambda storage, loc: storage.cuda(device))
        model_pretrain_en = PretrainClass(**config_pretrain).to(device)
        model_pretrain_en.load_state_dict(pt["model"])
        model_pretrain_en.eval()
        del pt
            
    # pretrain class decoder
    if pretrain:
        pt = torch.load(pretrain_path_de, map_location=lambda storage, loc: storage.cuda(device))
        model_pretrain_de = PretrainClass(**config_pretrain).to(device)
        model_pretrain_de.load_state_dict(pt["model"])
        model_pretrain_de.eval()
        del pt

    with torch.no_grad():
        if pretrain:
            cls = torch.tensor([[len(vocab)]]*X.size(0)).to(device)
            X_e = model_pretrain_en.get_output(torch.cat((cls, X), dim=1)) if "n_species" in config_pretrain else model_pretrain_en.get_output(X)
            X_e = torch.transpose(X_e, 1, 2)
            path_w = './test_w.txt'
            
        else:
            X_e = X

        if mode == MODE_GREEDY:
            out = model.infer_greedily(X_e, X, Y_sp, X.size(1), [17, 19, 25])
        elif mode == MODE_BEAM:
            outs = []
            for x_e, x, y_sp in zip(torch.split(X_e, batch_size), torch.split(X, batch_size), torch.split(Y_sp, batch_size)):
                x_e = torch.tensor(x_e, dtype=torch.float32)
                x_e = x_e.cuda()

                out = model(x_e, x, y_sp)
                out = torch.transpose(out, 1, 2)
                #decoderを外して実装
                #if pretrain:
                    #out = model_pretrain_de.get_output_decoder(out)
                out = torch.argmax(out, dim=2)

                if out.size(1) < 700:
                    out = torch.cat((out.cpu(), torch.zeros(out.size(0), 700-out.size(1)).long()), dim=1)
                   
                outs.append(out.cpu())         
                
            out = torch.cat(outs)
            out = out
                
                
            #out = model_pretrain_de.get_output_decoder(torch.cat((cls, out), dim=1)) if "n_species" in config_pretrain else model_pretrain_de.get_output_decoder(out)       
    #print("出力")
    print(out)
    print(out.shape)

    #write_fna_faa(out.cpu(), "{}/finetune/gen_cnn_align_12lay".format(output_dir), "GEN")  
    write_fna_faa_2(out.cpu(), "{}/finetune/gen_for_8_8/gen_train_sak_sub_3".format(output_dir), "GEN")  