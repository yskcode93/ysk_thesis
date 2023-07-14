#!/usr/bin/env python
# coding: utf-8

# In[120]:


# module insatall
import torch
import torch.nn as nn
import torch.optim as opt
import torch.optim.lr_scheduler as lrs
import torch.utils.data as dat
import torch.nn.utils as utils
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

# train test split
from sklearn.model_selection import KFold, StratifiedKFold

# base package
import re
import glob
import os
import time
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
from preprocess import fasta_to_data_,fasta_to_data, fasta_to_data_2
from Bio import Align, SeqIO, pairwise2
from Bio.Align import substitution_matrices
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import fcluster
import math
from sklearn.model_selection import train_test_split
from analysis import gc_content, log_cai, rscu, optimize_codon
import pickle


def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


# In[96]:


path = './CDSdata/10_aa/10_aa_data/GCF_022809095.1_ASM2280909v1_translated_cds.faa'
id_list = []
seq_list = []
des_list = []

for record in SeqIO.parse(path, 'fasta'):
    id_part = record.id
    desc_part = record.description
    desc_part = str(desc_part)
    pattern = r'\[locus_tag=(.+?)\]'
    match = re.search(pattern, desc_part)
    if match:
        desc_part = match.group(1)
    
    id_list.append(id_part)
    #seq_list.append(seq)
    des_list.append(desc_part)
    #print(id_part)
    #print(seq)
    
#sub_d = dict(zip(id_list, seq_list))
sub_d = dict(zip(id_list, des_list))

with open('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/10_aa.pkl', 'wb') as f:
    pickle.dump(sub_d, f)


# In[99]:


f1 = pickle_load('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/1_aa.pkl')
f2 = pickle_load('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/2_aa.pkl')
f3 = pickle_load('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/3_aa.pkl')
f4 = pickle_load('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/4_aa.pkl')
f5 = pickle_load('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/5_aa.pkl')
f6 = pickle_load('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/6_aa.pkl')
f7 = pickle_load('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/7_aa.pkl')
f8 = pickle_load('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/8_aa.pkl')
f9 = pickle_load('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/9_aa.pkl')
f10 = pickle_load('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/10_aa.pkl')
f1.update(**f2,**f3,**f4,**f5,**f6,**f7,**f8,**f9,**f10)
with open('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/all_aa.pkl', 'wb') as f:
    pickle.dump(f1, f)


# In[109]:


path = './CDSdata/10_aa/10_nc_data/raw_data/GCF_022809095.1_ASM2280909v1_cds_from_genomic.fna'
id_list = []
des_list = []
seq_list = []
for record in SeqIO.parse(path, 'fasta'):
    desc_part = record.description
    desc_part = str(desc_part)
    pattern = r'\[locus_tag=(.+?)\]'
    match = re.search(pattern, desc_part)
    if match:
        desc_part = match.group(1)
    
    seq = record.seq
    seq = str(seq)
    seq = seq .replace('Seq(','')
    seq  = seq .replace(')','')
    seq  = seq .replace('*','')
    
    des_list.append(desc_part)
    seq_list.append(seq)
    #print(id_part)
    #print(seq)
    
sub_d = dict(zip(des_list, seq_list))

with open('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/10_nc.pkl', 'wb') as f:
    pickle.dump(sub_d, f)


# In[111]:


f1 = pickle_load('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/1_nc.pkl')
f2 = pickle_load('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/2_nc.pkl')
f3 = pickle_load('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/3_nc.pkl')
f4 = pickle_load('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/4_nc.pkl')
f5 = pickle_load('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/5_nc.pkl')
f6 = pickle_load('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/6_nc.pkl')
f7 = pickle_load('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/7_nc.pkl')
f8 = pickle_load('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/8_nc.pkl')
f9 = pickle_load('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/9_nc.pkl')
f10 = pickle_load('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/10_nc.pkl')

f1.update(**f2,**f3,**f4,**f5,**f6,**f7,**f8,**f9,**f10)
len(f1)
with open('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/all_nc.pkl', 'wb') as f:
    pickle.dump(f1, f)


# In[112]:


df = pd.read_csv('./CDSdata/10_aa/res_sonic/runs/sonic_266238654_default_72cpus_ml05_ot_op/ortholog_relations/ortholog_pairs.sonic_266238654_default_72cpus_ml05_ot_op.tsv', sep='\t')
df


# In[116]:


all_locus = pickle_load('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/all_aa.pkl')
all_seq = pickle_load('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/all_nc.pkl')
df = pd.read_csv('./CDSdata/10_aa/res_sonic/runs/sonic_266238654_default_72cpus_ml05_ot_op/ortholog_relations/ortholog_pairs.sonic_266238654_default_72cpus_ml05_ot_op.tsv', sep='\t')

src_path = open('./CDSdata/10_aa/10_nc_data/src1.fna','a')
tgt_path = open('./CDSdata/10_aa/10_nc_data/tgt1.fna','a')

cnt = 0

for i in range(df.shape[0]):
    X_locus = all_locus[df.iat[i, 0]]
    Y_locus = all_locus[df.iat[i, 1]]
    
    if(X_locus in all_seq):
        if(Y_locus in all_seq):
            X_seq = all_seq[X_locus]
            Y_seq = all_seq[Y_locus]
            
            if(len(X_seq)<=2100 and len(Y_seq)<=2100):
                if('N' in X_seq or 'N' in Y_seq):
                    continue
                else:
                    cnt += 1
                    src_path.write(">srcdata_00{}\n".format(cnt))
                    src_path.write(X_seq)
                    src_path.write("\n")
                    tgt_path.write(">tgtdata_00{}\n".format(cnt))
                    tgt_path.write(Y_seq)
                    tgt_path.write("\n")
            else:
                continue
        else:
            continue
    else:
        continue        


# In[ ]:




