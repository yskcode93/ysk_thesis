#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
devices = "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
os.environ["CUDA_VISIBLE_DEVICES"] = devices
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12356'


# In[2]:


import torch.multiprocessing as mp
from convert import  MODE_GREEDY, MODE_BEAM, check
from convert import cross_validate_flocal,test_from_fasta_flocal,cross_validate_flocal_cnn, flocal_cnn_align,test_from_npy_flocal
from model import Converter, TransformerConverter, FastAutoregressiveConverter, CNN_converter_512, CNN_converter_512_8lay, CNN_converter_512_4lay
from model import CNN_converter_512_12lay,  CNN_converter_512_24lay, CNN_converter_one
from model2 import MultigMLP
import torch.nn as nn
from visualize import scatter_plot, scatter_plot_beam, METRIC_IDENTITY_NC, METRIC_IDENTITY_AA
from analysis import gc_content, log_cai, rscu, optimize_codon
from Bio import SeqIO
import numpy as np

# for transformer
config_transformer = {
    "n_tokens": 66,
    "seq_len": 700,
    "n_layers": 6,
    "n_heads": 2,
    "query_dimensions": 128,
    "value_dimensions": 128,
    "feed_forward_dimensions": 256,
    "attention_type": "full",
    "n_species": 34,
    "pretrain": True
}
n_epochs = 6000
batch_size = 4
lr = 1e-4 * 14
warmup = 0.1
use_apex = False
strain_ids = [
    
    22096, 15376, 22118, 22146, 8415, 21918, 20123, 452, 18655, 6750, 17659, 421, 22191, 21978, 12722, 17400,\
    15093, 20120, 20313, 20114, 22204, 19272, 17982, 19601, 21259, 22091, 1375, 10427, 18739, 18441, 22200, 22201, 22202, 22203
]
direction = 2
pretrain = True
pretrain_class = MultigMLP
config_pretrain = {
    "n_tokens": 67,
    "d_in": 512,
    "d_ffn": 1024,
    "max_len": 701,
    "n_layers": 32,
    "act": nn.Tanh(),
    "n_species": 278,
}
pretrain_path_en = "./Result/pretrain/pretrain_hiratani_data/pretrain_30.pt"
pretrain_path_de = "./Result/pretrain/protein_family_subtilis/protein_family_subtilis2.pt"

log_interval = 100
save_interval = 1000
output_dir = "./Result/pretrain/protein_family_stratified"
converter_class = CNN_converter_512_8lay
config = config_transformer

nprocs = len(devices.split(","))


# In[ ]:


mp.spawn(flocal_cnn_align, nprocs=nprocs, args=(nprocs, converter_class,  "{}/weight/weight_M2_7_11/cnn_align_10x10_pretrain/cnn_align_k15_main_10x10_6000.pt".format(output_dir),        n_epochs, batch_size, lr, warmup, use_apex, True,        strain_ids, [35], direction, MODE_BEAM,        pretrain, pretrain_class, config_pretrain, pretrain_path_en, pretrain_path_de,         log_interval, save_interval, output_dir))


# In[ ]:


mp.spawn(cross_validate_flocal_cnn, nprocs=nprocs, args=(nprocs, converter_class, "{}/weight/checkpoint_12000.pt".format(output_dir),        n_epochs, batch_size, lr, warmup, use_apex, False,        strain_ids, [35], direction, MODE_BEAM,        pretrain, pretrain_class, config_pretrain, pretrain_path_en, pretrain_path_de,         log_interval, save_interval, output_dir))


# In[3]:


mp.spawn(cross_validate_flocal, nprocs=nprocs, args=(nprocs, converter_class, config, "{}/weight/checkpoint_12000.pt".format(output_dir),        n_epochs, batch_size, lr, warmup, use_apex, False,        strain_ids, [35], direction, MODE_BEAM,        pretrain, pretrain_class, config_pretrain, pretrain_path_en, pretrain_path_de,         log_interval, save_interval, output_dir))


# In[4]:


from convert import test_from_fasta, MODE_GREEDY, MODE_BEAM
# generate sequences from source fasta file
test_from_fasta(converter_class, config, strain_ids, 22096, 2000, 1, MODE_BEAM,0,                pretrain, pretrain_class, config_pretrain, pretrain_path_en, pretrain_path_de,                output_dir, 100)


# In[3]:


test_from_npy_flocal(converter_class, config, strain_ids, 22096, 6000, 1, MODE_BEAM,0,                pretrain, pretrain_class, config_pretrain, pretrain_path_en, pretrain_path_de,                output_dir, 100)


# In[3]:


test_from_fasta_flocal(converter_class, config, strain_ids, 22096, 6000, 1, MODE_BEAM,0,                pretrain, pretrain_class, config_pretrain, pretrain_path_en, pretrain_path_de,                output_dir, 100)


# In[5]:


import glob
from preprocess import fasta_to_data
from file import GeneBatchLoader, sanitize, vocab, write_fna_faa
import torch

tr_thupath = "./Result/pretrain/protein_family_stratified/finetune/cluster_thuringiensis/cls_src_train_dis50_bad.fna"
tr_thupath = glob.glob(tr_thupath)
tr_subpath = "./Result/pretrain/protein_family_stratified/finetune/cluster_subtilis/cls_tgt_train_dis50_bad.fna"
tr_subpath = glob.glob(tr_subpath)
te_thupath = "./Result/pretrain/protein_family_stratified/finetune/cluster_thuringiensis/cls_src_test_dis50_bad.fna"
te_thupath = glob.glob(te_thupath)
te_subpath = "./Result/pretrain/protein_family_stratified/finetune/cluster_subtilis/cls_tgt_test_dis50_bad.fna"
te_subpath = glob.glob(te_subpath)
#Y_sp = torch.zeros(1447,1, dtype=torch.int64)

tr_X = fasta_to_data(tr_thupath, slen= 700)
tr_Y = fasta_to_data(tr_subpath, slen= 700)
te_X = fasta_to_data(te_thupath, slen= 700)
te_Y = fasta_to_data(te_subpath, slen= 700)

max_length = 701 if pretrain and "n_species" in config_pretrain else 700

if tr_X.size(1) < max_length:
    tr_X = torch.cat((tr_X, torch.zeros(tr_X.size(0), max_length-tr_X.size(1)).long()), dim=1)
if tr_Y.size(1) < max_length:
    tr_Y = torch.cat((tr_Y, torch.zeros(tr_Y.size(0), max_length-tr_Y.size(1)).long()), dim=1)
if te_X.size(1) < max_length:
    te_X = torch.cat((te_X, torch.zeros(te_X.size(0), max_length-te_X.size(1)).long()), dim=1)
if te_Y.size(1) < max_length:
    te_Y = torch.cat((te_Y, torch.zeros(te_Y.size(0), max_length-te_Y.size(1)).long()), dim=1)
    
print(tr_X)
print(tr_X.shape)
print(tr_X[0:,1:].shape)
    
write_fna_faa(te_X[0:,1:] if pretrain and "n_species" in config_pretrain else te_X, "{}/finetune/cluster_thuringiensis/cls_src_test_dis50_bad".format(output_dir), "SRC")
write_fna_faa(te_Y[0:,1:]  if pretrain and "n_species" in config_pretrain else te_Y, "{}/finetune/cluster_subtilis/cls_tgt_test_dis50_bad".format(output_dir), "TGT")
write_fna_faa(tr_X[0:,1:]  if pretrain and "n_species" in config_pretrain else tr_X, "{}/finetune/cluster_thuringiensis/cls_src_train_dis50_bad".format(output_dir), "SRC")
write_fna_faa(tr_Y[0:,1:]  if pretrain and "n_species" in config_pretrain else tr_Y, "{}/finetune/cluster_subtilis/cls_tgt_train_dis50_bad".format(output_dir), "TGT")


# In[4]:


#clusterセット　配列類似度
with open("{}/finetune/align_data_seq/tgt_train_cnn_align.fna".format(output_dir), "r") as f:
    tgt = list(SeqIO.parse(f, "fasta"))

with open("{}/finetune/gen_for_7_11/gen_train_cnn_align_k15_10x10_8lay.fna".format(output_dir), "r") as f:
    gen = list(SeqIO.parse(f, "fasta"))

with open("{}/finetune/align_data_seq/src_train_cnn_align.fna".format(output_dir), "r") as f:
    src = list(SeqIO.parse(f, "fasta"))

scatter_plot(src, tgt, gen, "{}/beam_search_nc_train_align_k15_10x10_8lay.png".format(output_dir),      metric=METRIC_IDENTITY_NC)


# In[5]:


#clusterセット　アミノ酸類似度
with open("{}/finetune/align_data_seq/tgt_train_cnn_align.faa".format(output_dir), "r") as f:
    tgt = list(SeqIO.parse(f, "fasta"))

with open("{}/finetune/gen_for_7_11/gen_train_cnn_align_k15_10x10_8lay.faa".format(output_dir), "r") as f:
    gen = list(SeqIO.parse(f, "fasta"))

with open("{}/finetune/align_data_seq/src_train_cnn_align.faa".format(output_dir), "r") as f:
    src = list(SeqIO.parse(f, "fasta"))

scatter_plot(src, tgt, gen, "{}/beam_search_aa_train_align_k15_10x10_8lay.png".format(output_dir),      metric=METRIC_IDENTITY_AA)


# In[ ]:


#clusterセット　配列類似度
with open("{}/finetune/align_data_seq/tgt_cnn_align.fna".format(output_dir), "r") as f:
    tgt = list(SeqIO.parse(f, "fasta"))

with open("{}/finetune/gen_for_4_24/gen_cnn_align_three_pretrain_1_2_3_threetoone.fna".format(output_dir), "r") as f:
    gen = list(SeqIO.parse(f, "fasta"))

with open("{}/finetune/align_data_seq/src_cnn_align.fna".format(output_dir), "r") as f:
    src = list(SeqIO.parse(f, "fasta"))

scatter_plot(src, tgt, gen, "{}/beam_search_nc_cnn_three_pretrain_1_2_3_threetoone.png".format(output_dir),      metric=METRIC_IDENTITY_NC)


# In[ ]:


#clusterセット　アミノ酸類似度
with open("{}/finetune/align_data_seq/tgt_cnn_align.faa".format(output_dir), "r") as f:
    tgt = list(SeqIO.parse(f, "fasta"))

with open("{}/finetune/gen_for_4_24/gen_cnn_align_three_pretrain_1_2_3_threetoone.faa".format(output_dir), "r") as f:
    gen = list(SeqIO.parse(f, "fasta"))

with open("{}/finetune/align_data_seq/src_cnn_align.faa".format(output_dir), "r") as f:
    src = list(SeqIO.parse(f, "fasta"))

scatter_plot(src, tgt, gen, "{}/beam_search_aa_cnn_three_pretrain_1_2_3_threetoone.png".format(output_dir),      metric=METRIC_IDENTITY_AA)


# In[5]:


#通常データセット　配列類似度
for i in range(1,2):
    with open("{}/finetune/tgt_{}.fna".format(output_dir, i), "r") as f:
        tgt = list(SeqIO.parse(f, "fasta"))

    with open("{}/finetune/gen_{}.fna".format(output_dir, i), "r") as f:
        gen = list(SeqIO.parse(f, "fasta"))

    with open("{}/finetune/src_{}.fna".format(output_dir, i), "r") as f:
        src = list(SeqIO.parse(f, "fasta"))
        
    #print(np.exp([log_cai(str(x.seq), ref) for x in gen]).mean())
    #print(np.array([gc_content(str(x.seq)) for x in gen]).mean())
    scatter_plot(src, tgt, gen, "{}/beam_search_nc_notest{}.png".format(output_dir, i),      metric=METRIC_IDENTITY_NC)


# In[6]:


#通常データセット アミノ酸類似度

for i in range(1,2):
    with open("{}/finetune/tgt_{}.faa".format(output_dir, i), "r") as f:
        tgt = list(SeqIO.parse(f, "fasta"))

    with open("{}/finetune/gen_{}.faa".format(output_dir, i), "r") as f:
        gen = list(SeqIO.parse(f, "fasta"))

    with open("{}/finetune/src_{}.faa".format(output_dir, i), "r") as f:
        src = list(SeqIO.parse(f, "fasta"))
        
    #print(np.exp([log_cai(str(x.seq), ref) for x in gen]).mean())
    #print(np.array([gc_content(str(x.seq)) for x in gen]).mean())
    scatter_plot(src, tgt, gen, "{}/beam_search_aa_notest{}.png".format(output_dir, i),      metric=METRIC_IDENTITY_AA)


# In[6]:


from convert import test_from_train, test_from_train_flocal, MODE_GREEDY, MODE_BEAM
# generate sequences from source fasta file
test_from_train_flocal(converter_class, config, strain_ids, 22096, 6000, 1, MODE_BEAM,6,                pretrain, pretrain_class, config_pretrain, pretrain_path_en, pretrain_path_de,                output_dir, 100)


# In[ ]:


#clusterセット　train配列類似度
with open("{}/finetune/tgt_train_1.fna".format(output_dir), "r") as f:
    tgt = list(SeqIO.parse(f, "fasta"))

with open("{}/finetune/gen_train_cnn_512_24lay_rev_enno.fna".format(output_dir), "r") as f:
    gen = list(SeqIO.parse(f, "fasta"))

with open("{}/finetune/src_train_1.fna".format(output_dir), "r") as f:
    src = list(SeqIO.parse(f, "fasta"))

scatter_plot(src, tgt, gen, "{}/beam_search_nc_cnn_train_512_24lay_rev_6000_enno.png".format(output_dir),      metric=METRIC_IDENTITY_NC)


# In[ ]:


#clusterセット　trainアミノ酸類似度
with open("{}/finetune/tgt_train_1.faa".format(output_dir), "r") as f:
    tgt = list(SeqIO.parse(f, "fasta"))

with open("{}/finetune/gen_train_cnn_512_24lay_rev_enno.faa".format(output_dir), "r") as f:
    gen = list(SeqIO.parse(f, "fasta"))

with open("{}/finetune/src_train_1.faa".format(output_dir), "r") as f:
    src = list(SeqIO.parse(f, "fasta"))

scatter_plot(src, tgt, gen, "{}/beam_search_aa_cnn_train_512_24lay_rev_6000_enno.png".format(output_dir),      metric=METRIC_IDENTITY_AA)


# In[8]:


#通常データセット　train配列類似度

for i in range(1,2):
    with open("{}/finetune/tgt_train_{}.fna".format(output_dir, i), "r") as f:
        tgt = list(SeqIO.parse(f, "fasta"))

    with open("{}/finetune/gen_train_{}.fna".format(output_dir, i), "r") as f:
        gen = list(SeqIO.parse(f, "fasta"))

    with open("{}/finetune/src_train_{}.fna".format(output_dir, i), "r") as f:
        src = list(SeqIO.parse(f, "fasta"))
        
    #print(np.exp([log_cai(str(x.seq), ref) for x in gen]).mean())
    #print(np.array([gc_content(str(x.seq)) for x in gen]).mean())
    scatter_plot(src, tgt, gen, "{}/beam_search_nc_train_notest{}.png".format(output_dir, i),      metric=METRIC_IDENTITY_NC)


# In[9]:


#通常データセット　アミノ酸類似度

for i in range(1,2):
    with open("{}/finetune/tgt_train_{}.faa".format(output_dir, i), "r") as f:
        tgt = list(SeqIO.parse(f, "fasta"))

    with open("{}/finetune/gen_train_{}.faa".format(output_dir, i), "r") as f:
        gen = list(SeqIO.parse(f, "fasta"))

    with open("{}/finetune/src_train_{}.faa".format(output_dir, i), "r") as f:
        src = list(SeqIO.parse(f, "fasta"))
        
    #print(np.exp([log_cai(str(x.seq), ref) for x in gen]).mean())
    #print(np.array([gc_content(str(x.seq)) for x in gen]).mean())
    scatter_plot(src, tgt, gen, "{}/beam_search_aa_train_notest{}.png".format(output_dir, i),      metric=METRIC_IDENTITY_AA)


# In[ ]:




