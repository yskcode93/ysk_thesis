#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[46]:


df = pd.read_csv('./CDSdata/12_aa/res_sonic/runs/sonic_286236257_default_72cpus_ml05_ot_op/ortholog_groups/ortholog_groups.tsv', sep='\t')
src_ID = pickle_load('./CDSdata/12_aa/test_srcID.pkl')
tgt_ID = pickle_load('./CDSdata/12_aa/test_tgtID.pkl')


# In[47]:


df 


# In[9]:


src_list = []
tgt_list = []
for i in range(len(src_ID)):
    check_w = src_ID[i]
    for j in range(df.shape[0]):
        if(check_w in df.iat[j,8]):
            src_list.append(j)

for i in range(len(tgt_ID)):
    check_w = tgt_ID[i]
    for j in range(df.shape[0]):
        if(check_w in df.iat[j,4]):
            tgt_list.append(j)         

group_list = src_list + tgt_list
group_list = set(group_list)
group_list


# In[15]:


f = open('./CDSdata/12_aa/test_group.pkl',"rb")
list_row = pickle.load(f)
len(list_row)


# In[29]:


test_group = list_row
non_test_list = []
for i in range(df.shape[0]):
    if(i in list_row):
        continue
    else:
        for j in range (df.shape[1]):
            if(j%2==0 and j!=0 and j != 2 and j != 4 and j!= 8):
                s = df.iat[i,j]
                s = str(s)
                l = s.split(',')
                non_test_list = non_test_list + l

res_list = []
for i in range(len(non_test_list)):
    if(non_test_list[i]=='*'):
        continue
    else:
        res_list.append(non_test_list[i])
        
f = open('./CDSdata/12_aa/no_testID.pkl', 'wb')
list_row = res_list
pickle.dump(list_row, f)


# In[2]:


pair_df = pd.read_csv('./CDSdata/12_aa/res_sonic/runs/sonic_286236257_default_72cpus_ml05_ot_op/ortholog_relations/ortholog_pairs.sonic_286236257_default_72cpus_ml05_ot_op.tsv', sep='\t')
pair_df.shape


# In[48]:


all_locus = pickle_load('./CDSdata/10_aa/10_aa_data/dics_10aa_id_locus/all_aa.pkl')
all_seq = pickle_load('./CDSdata/10_aa/10_nc_data/dics_10nc_locus_seq/all_nc.pkl')
pair_df = pd.read_csv('./CDSdata/12_aa/res_sonic/runs/sonic_286236257_default_72cpus_ml05_ot_op/ortholog_relations/ortholog_pairs.sonic_286236257_default_72cpus_ml05_ot_op.tsv', sep='\t')

#src_path = open('./CDSdata/12_aa/nc_data/src.fna','a')
#tgt_path = open('./CDSdata/12_aa/nc_data/tgt.fna','a')

f = open('./CDSdata/12_aa/no_testID.pkl',"rb")
list_row = pickle.load(f)

cnt = 0
i_list = []
for i in range(pair_df.shape[0]):
    if(pair_df.iat[i, 0] in list_row and pair_df.iat[i, 1] in list_row):
        X_locus = all_locus[pair_df.iat[i, 0]]
        Y_locus = all_locus[pair_df.iat[i, 1]]
        if(X_locus in all_seq and Y_locus in all_seq):
            X_seq = all_seq[X_locus]
            Y_seq = all_seq[Y_locus]
                
            if(len(X_seq)<=2100 and len(Y_seq)<=2100):
                if('N' in X_seq or 'N' in Y_seq):
                    continue    
                if(len(X_seq)>len(Y_seq)):
                    if(len(X_seq)*0.97>len(Y_seq)):
                        continue
                elif(len(X_seq)<len(Y_seq)):
                    if(len(Y_seq)*0.97>len(X_seq)):
                        continue
                else:
                    cnt += 1
                    src_path = open('./CDSdata/12_aa/nc_data/src1.fna', 'a')
                    src_path.write(">srcdata_00{}\n".format(cnt))
                    src_path.write(X_seq)
                    src_path.write("\n")
                    src_path.close()
                    
                    tgt_path = open('./CDSdata/12_aa/nc_data/tgt1.fna', 'a')
                    tgt_path.write(">tgtdata_00{}\n".format(cnt))
                    tgt_path.write(Y_seq)
                    tgt_path.write("\n")
                    tgt_path.close()


# In[ ]:




