#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
devices = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
os.environ["CUDA_VISIBLE_DEVICES"] = devices
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12361'


# In[2]:


import torch.multiprocessing as mp
from pretrain import train, train_, train_fam_cls, train_fam_cls_2, train_fam_cls_align
from model import TransformerEncoder, SPgMLP
from model2 import MultigMLP, gMLP
import torch.nn as nn
# for transformer
config_transformer = {
    "n_tokens": 66,
    "seq_len": 700,
    "n_layers": 12,
    "n_heads": 2,
    "query_dimensions": 128,
    "value_dimensions": 128,
    "feed_forward_dimensions": 256,
    "attention_type": "linear",
    "n_species": 34
}
# for gMLP
config_gmlp = {
    "num_tokens": 66,
    "dim": 256,
    "depth": 32,
    "seq_len": 701,
    "heads": 1,
    "ff_mult": 2,
    "attn_dim": None,
    "prob_survival": 1.,
    "causal": False,
    "circulant_matrix": True,
    "shift_tokens": 0,
    "act": nn.Tanh(),
    "n_species": 278
}
# for gmlp
config_gmlp = {
    "n_tokens": 67,
    "d_in": 512,
    "d_ffn": 1024,
    "max_len": 701,
    "n_layers": 32,
    "act": nn.Tanh(),
    "n_species": 278,
}
n_epochs = 6000
batch_size = 16
lr = 2e-4
warmup = 0.1
use_apex = True
strain_ids = [
    22096, 15376, 22118, 22146, 8415, 21918, 20123, 452, 18655, 6750, 17659, 421, 22191, 21978, 12722, 17400,\
    15093, 20120, 20313, 20114, 22204, 19272, 17982, 19601, 21259, 22091, 1375, 10427, 18739, 18441, 22200, 22201, 22202, 22203
]
strain_ids = [
    22096, 15376, 22118, 22146, 8415
]
log_interval = 100
save_interval = 1000
log = "./Result/pretrain/tmp.log"
checkpoint_path = "./Result/pretrain/tmp.pt"
pretrain_class = MultigMLP
config = config_gmlp

nprocs = len(devices.split(","))


# In[ ]:


mp.spawn(train_fam_cls_align, nprocs=nprocs, args=(nprocs, pretrain_class, config, n_epochs, batch_size, lr, warmup, use_apex,    log_interval, save_interval, False, log, checkpoint_path))


# In[ ]:




