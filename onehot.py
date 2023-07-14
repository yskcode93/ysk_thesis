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
from file import GeneBatchLoader, sanitize, vocab, write_fna_faa
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

def one_hot(categories, string):
    encoding = np.zeros((len(categories), len(string)))
    for idx, char in enumerate(string):
        encoding[categories.index(char), idx] = 1
    return encoding

def featurize(genome):
    genome = genome.replace('Seq(','')
    genome = genome.replace(')','')
    genome = re.split('(...)', str(genome))[1:-1:2]
    sequence = one_hot(["TTT", "TTC", "TTA", "TTG", "CTT", "CTC", "CTA", "CTG","ATT", "ATC", "ATA","GTT", "GTC", "GTA", "GTG",
    "TCT", "TCC", "TCA", "TCG", "AGT", "AGC","CCT", "CCC", "CCA", "CCG","ACT", "ACC", "ACA", "ACG","GCT", "GCC", "GCA", "GCG",
    "TAT", "TAC","CAT", "CAC","CAA", "CAG","AAT", "AAC","AAA", "AAG","GAT", "GAC","GAA", "GAG","TGT", "TGC","CGT", "CGC", "CGA", "CGG", 
    "AGA","AGG","GGT", "GGC", "GGA", "GGG","ATG","TGG"], genome)
    if(sequence.shape[1]<700):
        addition_dim = 700 - sequence.shape[1]
        pud = np.zeros((len(categories), addition_dim))
        sequence = np.hstack((sequence, pud))
    return sequence
