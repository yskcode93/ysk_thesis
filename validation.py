from os import write
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as utils

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
import torch.multiprocessing as mp
# apex pytorch extension
from apex.parallel import DistributedDataParallel as DDP_APEX
from apex import amp

from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from .analysis import translation_map, alignment, nc_similarity, amino_to_codon, LEVEL_AMINO

from base.pretrain import SPgMLP, dropout_layers
from g_mlp_pytorch import gMLP

import re
import os

import numpy as np

import itertools

# コドン配列からアミノ酸配列に変換
def to_aa(codon):
    return "".join(map(lambda x: translation_map[x] if x in translation_map else "", re.split('(...)', codon)[1:-1:2]))

# 文字列へ変換
def to_seq(x, vocab):
    return np.apply_along_axis(lambda x: ''.join(x), 0, np.array(vocab)[x].T)

# ファイルへ保存
def to_file(seq, filename, description, prefix="GE", gene_id=None):
    records = []

    if gene_id:
        record = SeqRecord(Seq(seq[0]), "{}_{:0=6}".format(prefix, gene_id), basename(filename).split("_")[0], description)
        records.append(record)
    else:
        for i, s in enumerate(seq):
            record = SeqRecord(Seq(s), "{}_{:0=6}".format(prefix, i+1), "contig_{:0=6}".format(i+1), description)
            records.append(record)

    with open(filename, "w") as f:
        SeqIO.write(records, f, "fasta")

# 平均配列類似度
def average_identity(X_nc, X_aa, Y_nc, Y_aa):
    identities = np.array([])
    for x_nc, x_aa, y_nc, y_aa in zip(X_nc, X_aa, Y_nc, Y_aa):
        identities = np.append(identities, nc_similarity(*amino_to_codon(x_nc, y_nc, alignment(x_aa, y_aa, LEVEL_AMINO))))

    return identities.mean()

# position-wise converter including codon embedding as inputs
class PositionwiseConverterConcat(nn.Module):
    def __init__(self, n_units):
        super(PositionwiseConverterConcat, self).__init__()
        
        d_in = 512
        
        self.embedding = nn.Embedding(66, d_in)
        
        self.position_wise_mlp = nn.Sequential(
            nn.Linear(d_in*2, n_units),
            nn.LayerNorm(n_units),
            nn.ELU(),
            #nn.Dropout(0.2),
            nn.Linear(n_units, 66)
        )
        
    def forward(self, x, ref):
        embedding = self.embedding(ref)
        x = torch.cat((x, embedding), dim=2)
        x = self.position_wise_mlp(x)
        
        return x

class PositionwiseConverter(nn.Module):
    def __init__(self):
        super(PositionwiseConverter, self).__init__()
        
        d_in = 512
        
        self.position_wise_mlp = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.LayerNorm(d_in),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(d_in, d_in),
            nn.LayerNorm(d_in),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(d_in, 66)
        )
        
    def forward(self, x, ref):
        x = self.position_wise_mlp(x)
        
        return x

# 畳み込みベースの変換器
class ConvolutionalConverter(nn.Module):
    def __init__(self, kernel_size, n_filters):
        super(ConvolutionalConverter, self).__init__()
        
        d_in = 512
        
        self.embedding = nn.Embedding(66, d_in)
        
        self.conv = nn.Sequential(
            nn.Conv1d(d_in*2, n_filters, kernel_size, padding=kernel_size//2),
            nn.InstanceNorm1d(n_filters),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Conv1d(n_filters, n_filters, kernel_size, padding=kernel_size//2),
            nn.InstanceNorm1d(n_filters),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Conv1d(n_filters, 66, 1)
        )
        
    def forward(self, x, ref):
        embedding = self.embedding(ref)
        x = torch.cat((x, embedding), dim=2)
        x = x.permute(0,2,1)
        x = self.conv(x)
        x = x.permute(0,2,1)
        
        return x

# gmlp
class Converter(nn.Module):
    def __init__(self, n_species):
        super(Converter, self).__init__()

        self.layers = SPgMLP(
            dim = 512,
            depth = 6,
            seq_len = 700,
            num_tokens = 66,
            ff_mult = 2,
            heads = 1,
            circulant_matrix = True,
            act = nn.Tanh(),
            prob_survival = 0.99,
            n_species=n_species
        )
        
    def forward(self, x_e, x, sp):
        
        return self.layers(x, sp)

class ConverterWithContext(gMLP):
    def __init__(
        self,
        *,
        num_tokens = None,
        dim,
        depth,
        seq_len,
        species_embedding,
        heads = 1,
        ff_mult = 4,
        attn_dim = None,
        prob_survival = 1.,
        causal = False,
        circulant_matrix = False,
        shift_tokens = 0,
        act = nn.Identity()
    ):
        super(ConverterWithContext, self).__init__(
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

        self.register_buffer("species_embedding", species_embedding)

    def forward(self, x_e, x, label):
        x = x_e + self.to_embed(x) + self.species_embedding[label]
        layers = self.layers if not self.training else dropout_layers(self.layers, self.prob_survival)
        out = nn.Sequential(*layers)(x)
        return self.to_logits(out)

class CNN(nn.Module):
    def __init__(self, n_species, species_embedding=None, has_context=True):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(512, 1024, 5, padding=2),
            nn.InstanceNorm1d(1024),
            nn.ELU(),
            nn.Conv1d(1024, 512, 5, padding=2),
            nn.InstanceNorm1d(512),
            nn.ELU(),
            nn.Conv1d(512, 66, 1)
        )

        if has_context:
            self.register_buffer("species_embedding", species_embedding)
        else:
            self.species_embedding = nn.Embedding(n_species, 512)
        self.codon_embedding = nn.Embedding(66, 512)
        self.has_context = has_context

    def forward(self, x_e, x, sp):
        if self.has_context:
            x = x_e + self.codon_embedding(x) + self.species_embedding[sp]
        else:
            x = self.codon_embedding(x) + self.species_embedding(sp)

        x = x.permute(0,2,1)

        return self.conv(x).permute(0,2,1)

# 訓練セットと評価セットをランダムに分割
def random_cross_validate(ConverterClass, config, input, input_, target, target_id, n_epochs, batch_size, device=0, log_interval=100, parent_dir="./Result"):
    # シード値
    torch.manual_seed(0)
    # フォールドの分割
    tgt_id_set = np.unique(target_id)
    kf = KFold(random_state=0, shuffle=True)

    identity_val = 0.0

    codons = itertools.product("ATGC", repeat=3)
    codons = ["".join(c) for c in codons]
    vocab = ["<pad>"] + codons + ["<msk>"]

    input = input.cpu()
    input_ = input_.cpu()
    target = target.cpu()

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(tgt_id_set)):
        tgt_seq_test = tgt_id_set[test_idx]
        test_idx = np.isin(target_id, tgt_seq_test)
        train_idx = ~ test_idx
        input_train, input_test = input[train_idx], input[test_idx]
        input_train_, input_test_ = input_[train_idx], input_[test_idx]
        target_train, target_test = target[train_idx], target[test_idx]

        dataset = TensorDataset(input_train, input_train_, target_train)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        # モデルの定義
        model = ConverterClass(**config).to(device)
        optimizer = optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()

        # 学習
        for i in tqdm(range(1, n_epochs + 1)):
            for j, (x, x_, y) in enumerate(loader):
                optimizer.zero_grad()
                out = model(x.to(device), x_.to(device))
                
                loss = criterion(out.permute(0,2,1), y.to(device))
                loss.backward()
                
                optimizer.step()

            if i==1 or i%log_interval==0:
                trans = str.maketrans({
                    '<': '',
                    '>': '',
                    'p': '',
                    'a': '',
                    'd': '',
                    's': '',
                    'o': '',
                    'e': '',
                    'm': '',
                    'k': ''
                })
                # 訓練データでの変換
                with torch.no_grad():
                    out = model(input_train.to(device), input_train_.to(device))

                    src_str = to_seq(input_train_.cpu(), vocab)
                    tgt_str = to_seq(target_train.cpu(), vocab)
                    gen_str = to_seq(torch.argmax(out, dim=2).cpu(), vocab)

                    src_str = [s.translate(trans) for s in src_str]
                    tgt_str = [t.translate(trans) for t in tgt_str]
                    gen_str = [g.translate(trans) for g in gen_str]
                    
                    src_aa = [to_aa(s) for s in src_str]
                    tgt_aa = [to_aa(t) for t in tgt_str]
                    gen_aa = [to_aa(g) for g in gen_str]
                    # 出力の保存
                    to_file(src_str, "{}/train/src_{}_{}.fna".format(parent_dir, fold_idx+1, i), "", "SRC")
                    to_file(tgt_str, "{}/train/tgt_{}_{}.fna".format(parent_dir, fold_idx+1, i), "", "TGT")
                    to_file(gen_str, "{}/train/gen_{}_{}.fna".format(parent_dir, fold_idx+1, i), "","GEN")
                    to_file(src_aa, "{}/train/src_{}_{}.faa".format(parent_dir, fold_idx+1, i), "", "SRC")
                    to_file(tgt_aa, "{}/train/tgt_{}_{}.faa".format(parent_dir, fold_idx+1, i), "", "TGT")
                    to_file(gen_aa, "{}/train/gen_{}_{}.faa".format(parent_dir, fold_idx+1, i), "","GEN")

                # 検証データでの変換
                with torch.no_grad():
                    out = model(input_test.to(device), input_test_.to(device))

                    src_str = to_seq(input_test_.cpu(), vocab)
                    tgt_str = to_seq(target_test.cpu(), vocab)
                    gen_str = to_seq(torch.argmax(out, dim=2).cpu(), vocab)

                    src_str = [s.translate(trans) for s in src_str]
                    tgt_str = [t.translate(trans) for t in tgt_str]
                    gen_str = [g.translate(trans) for g in gen_str]
                    src_aa = [to_aa(s) for s in src_str]
                    tgt_aa = [to_aa(t) for t in tgt_str]
                    gen_aa = [to_aa(g) for g in gen_str]

                    #identity_aa = average_identity(tgt_aa, gen_aa)
                    #identity_nc = average_identity(tgt_str, tgt_aa, gen_str, gen_aa)
                    # 出力の保存
                    to_file(src_str, "{}/test/src_{}_{}.fna".format(parent_dir, fold_idx+1, i), "", "SRC")
                    to_file(tgt_str, "{}/test/tgt_{}_{}.fna".format(parent_dir, fold_idx+1, i), "", "TGT")
                    to_file(gen_str, "{}/test/gen_{}_{}.fna".format(parent_dir, fold_idx+1, i), "","GEN")
                    to_file(src_aa, "{}/test/src_{}_{}.faa".format(parent_dir, fold_idx+1, i), "", "SRC")
                    to_file(tgt_aa, "{}/test/tgt_{}_{}.faa".format(parent_dir, fold_idx+1, i), "", "TGT")
                    to_file(gen_aa, "{}/test/gen_{}_{}.faa".format(parent_dir, fold_idx+1, i), "","GEN")

            if i==n_epochs:
               identity_val += average_identity(tgt_str, tgt_aa, gen_str, gen_aa)
    return identity_val / (fold_idx + 1)

def write_fna_faa(x, basename, prefix):
    # 語彙  
    codons = itertools.product("ATGC", repeat=3)
    codons = ["".join(c) for c in codons]
    vocab = ["<pad>"] + codons + ["<msk>"]

    trans = str.maketrans({
        '<': '',
        '>': '',
        'p': '',
        'a': '',
        'd': '',
        's': '',
        'o': '',
        'e': '',
        'm': '',
        'k': ''
    })

    x_str = to_seq(x.cpu(), vocab)
    x_str = [x.translate(trans) for x in x_str]
    x_aa = [to_aa(x) for x in x_str]

    to_file(x_str, "{}.fna".format(basename), "", prefix)
    to_file(x_aa, "{}.faa".format(basename), "", prefix)

# 菌種レベルでの多対多の学習
MODE_RANDOM = 0
MODE_SP_ID = 1
MODE_SP_OD = 2
MODE_PFLEVEL_OD = 3
def many_to_many(Model, config, dataset, X_e, strain_ids, n_epochs, batch_size, n_samples=100, mode=MODE_SP_ID, log_interval=100, parent_dir="./Result"):
    # シード値
    torch.manual_seed(0)

    # to cpu
    dataset = torch.load("multisp_no_indel.pt")
    X = dataset["X"]
    Y = dataset["Y"]
    X_id = dataset["X_id"].numpy()
    Y_id = dataset["Y_id"].numpy()
    X_sp = dataset["X_sp"].numpy()
    Y_sp = dataset["Y_sp"].numpy()

    X_e = X_e.cpu()

    # map to index
    to_idx = np.vectorize(lambda x: strain_ids.index(x))
    X_sp = to_idx(X_sp)
    Y_sp = to_idx(Y_sp)

    # フォールドの分割
    if mode == MODE_RANDOM:
        kf = KFold(n_splits=5, random_state=0, shuffle=True)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            X_e_train, X_e_test = X_e[train_idx], X_e[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            Y_sp_train, Y_sp_test = Y_sp[train_idx], Y_sp[test_idx]

            dataset = TensorDataset(X_e_train, X_train, Y_train, torch.tensor(Y_sp_train).unsqueeze(1))
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            # モデルの定義
            model = Model(**config).cuda()
            optimizer = optim.AdamW(model.parameters())
            criterion = nn.CrossEntropyLoss()

            # 学習
            for i in tqdm(range(1, n_epochs + 1)):
                for j, (x_e, x, y, y_sp) in enumerate(loader):
                    optimizer.zero_grad()
                    out = model(x_e.cuda(), x.cuda(), y_sp.cuda())
                    
                    loss = criterion(out.permute(0,2,1), y.cuda())
                    loss.backward()
                    
                    optimizer.step()

                if i==1 or i%log_interval==0:
                    # 訓練データでの変換
                    with torch.no_grad():
                        # ランダムに100本をサンプリング
                        idx = torch.randint(X_train.size(0), size=(n_samples,))
                        x_e = X_e_train[idx].cuda()
                        x = X_train[idx].cuda()
                        y = Y_train[idx].cuda()
                        y_sp = torch.tensor(Y_sp_train[idx]).unsqueeze(1).cuda()
                        out = model(x_e, x, y_sp)

                        write_fna_faa(x, "{}/train/src_{}_{}".format(parent_dir, fold_idx+1, i), "SRC")
                        write_fna_faa(y, "{}/train/tgt_{}_{}".format(parent_dir, fold_idx+1, i), "TGT")
                        write_fna_faa(torch.argmax(out, dim=2).cpu(), "{}/train/gen_{}_{}".format(parent_dir, fold_idx+1, i), "GEN")

                    # 検証データでの変換
                    with torch.no_grad():
                        # ランダムに100本をサンプリング
                        idx = torch.randint(X_test.size(0), size=(n_samples,))
                        x_e = X_e_test[idx].cuda()
                        x = X_test[idx].cuda()
                        y = Y_test[idx].cuda()
                        y_sp = torch.tensor(Y_sp_test[idx]).unsqueeze(1).cuda()
                        out = model(x_e, x, y_sp)

                        write_fna_faa(x, "{}/test/src_{}_{}".format(parent_dir, fold_idx+1, i), "SRC")
                        write_fna_faa(y, "{}/test/tgt_{}_{}".format(parent_dir, fold_idx+1, i), "TGT")
                        write_fna_faa(torch.argmax(out, dim=2).cpu(), "{}/train/gen_{}_{}".format(parent_dir, fold_idx+1, i), "GEN")

    elif mode == MODE_SP_ID:
        sp_pairs = np.hstack(X_sp[:, np.newaxis], Y_sp[:, np.newaxis])
        unique_sp_pairs = np.unique(sp_pairs, axis=1)
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(unique_sp_pairs)):
            pairs_test = unique_sp_pairs[test_idx]
            test_idx = sp_pairs == pairs_test
            train_idx = ~test_idx
            
            X_train, X_test = X[train_idx], X[test_idx]
            X_e_train, X_e_test = X_e[train_idx], X_e[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            Y_sp_train, Y_sp_test = Y_sp[train_idx], Y_sp[test_idx]

            dataset = TensorDataset(X_e_train, X_train, Y_train, torch.tensor(Y_sp_train).unsqueeze(1))
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            # モデルの定義
            model = Model(**config).cuda()
            optimizer = optim.AdamW(model.parameters())
            criterion = nn.CrossEntropyLoss()

            # 学習
            for i in tqdm(range(1, n_epochs + 1)):
                for j, (x_e, x, y, y_sp) in enumerate(loader):
                    optimizer.zero_grad()
                    out = model(x_e.cuda(), x.cuda(), y_sp.cuda())
                    
                    loss = criterion(out.permute(0,2,1), y.cuda())
                    loss.backward()
                    
                    optimizer.step()

                if i==1 or i%log_interval==0:
                    # 訓練データでの変換
                    with torch.no_grad():
                        # ランダムに100本をサンプリング
                        idx = torch.randint(X_train.size(0), size=(n_samples,))
                        x_e = X_e_train[idx].cuda()
                        x = X_train[idx].cuda()
                        y = Y_train[idx].cuda()
                        y_sp = torch.tensor(Y_sp_train[idx]).unsqueeze(1).cuda()
                        out = model(x_e, x, y_sp)

                        write_fna_faa(x, "{}/train/src_{}_{}".format(parent_dir, fold_idx+1, i), "SRC")
                        write_fna_faa(y, "{}/train/tgt_{}_{}".format(parent_dir, fold_idx+1, i), "TGT")
                        write_fna_faa(torch.argmax(out, dim=2).cpu(), "{}/train/gen_{}_{}".format(parent_dir, fold_idx+1, i), "GEN")

                    # 検証データでの変換
                    with torch.no_grad():
                        # ランダムに100本をサンプリング
                        idx = torch.randint(X_test.size(0), size=(n_samples,))
                        x_e = X_e_test[idx].cuda()
                        x = X_test[idx].cuda()
                        y = Y_test[idx].cuda()
                        y_sp = torch.tensor(Y_sp_test[idx]).unsqueeze(1).cuda()
                        out = model(x_e, x, y_sp)

                        write_fna_faa(x, "{}/test/src_{}_{}".format(parent_dir, fold_idx+1, i), "SRC")
                        write_fna_faa(y, "{}/test/tgt_{}_{}".format(parent_dir, fold_idx+1, i), "TGT")
                        write_fna_faa(torch.argmax(out, dim=2).cpu(), "{}/train/gen_{}_{}".format(parent_dir, fold_idx+1, i), "GEN")

    elif mode == MODE_SP_OD:
        unique_sp = np.unique(X_sp)
        
        for fold_idx in range(5):
            _, sp_test = train_test_split(unique_sp, random_state=fold_idx, test_size=0.447)
            test_idx = np.logical_and(np.isin(X_sp, sp_test), np.isin(Y_sp, sp_test))
            train_idx = ~ test_idx

            X_train, X_test = X[train_idx], X[test_idx]
            X_e_train, X_e_test = X_e[train_idx], X_e[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            Y_sp_train, Y_sp_test = Y_sp[train_idx], Y_sp[test_idx]

            dataset = TensorDataset(X_e_train, X_train, Y_train, torch.tensor(Y_sp_train).unsqueeze(1))
            loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            # モデルの定義
            model = Model(**config).cuda()
            optimizer = optim.AdamW(model.parameters())
            criterion = nn.CrossEntropyLoss()

            # 学習
            for i in tqdm(range(1, n_epochs + 1)):
                for j, (x_e, x, y, y_sp) in enumerate(loader):
                    optimizer.zero_grad()
                    out = model(x_e.cuda(), x.cuda(), y_sp.cuda())
                    
                    loss = criterion(out.permute(0,2,1), y.cuda())
                    loss.backward()
                    
                    optimizer.step()

                if i==1 or i%log_interval==0:
                    # 訓練データでの変換
                    with torch.no_grad():
                        # ランダムに100本をサンプリング
                        idx = torch.randint(X_train.size(0), size=(n_samples,))
                        x_e = X_e_train[idx].cuda()
                        x = X_train[idx].cuda()
                        y = Y_train[idx].cuda()
                        y_sp = torch.tensor(Y_sp_train[idx]).unsqueeze(1).cuda()
                        out = model(x_e, x, y_sp)

                        write_fna_faa(x, "{}/train/src_{}_{}".format(parent_dir, fold_idx+1, i), "SRC")
                        write_fna_faa(y, "{}/train/tgt_{}_{}".format(parent_dir, fold_idx+1, i), "TGT")
                        write_fna_faa(torch.argmax(out, dim=2).cpu(), "{}/train/gen_{}_{}".format(parent_dir, fold_idx+1, i), "GEN")

                    # 検証データでの変換
                    with torch.no_grad():
                        # ランダムに100本をサンプリング
                        idx = torch.randint(X_test.size(0), size=(n_samples,))
                        x_e = X_e_test[idx].cuda()
                        x = X_test[idx].cuda()
                        y = Y_test[idx].cuda()
                        y_sp = torch.tensor(Y_sp_test[idx]).unsqueeze(1).cuda()
                        out = model(x_e, x, y_sp)

                        write_fna_faa(x, "{}/test/src_{}_{}".format(parent_dir, fold_idx+1, i), "SRC")
                        write_fna_faa(y, "{}/test/tgt_{}_{}".format(parent_dir, fold_idx+1, i), "TGT")
                        write_fna_faa(torch.argmax(out, dim=2).cpu(), "{}/train/gen_{}_{}".format(parent_dir, fold_idx+1, i), "GEN")

    elif mode == MODE_PFLEVEL_OD:
        pass

def get_sample_weights(ids):
    _, idx, counts = np.unique(ids, return_inverse=True, return_counts=True)
    weights = 1 / counts
    weights = weights / weights.sum()

    return weights[idx]

def encode_dataset(path, strain_ids):
    dataset = torch.load(path)
    X = dataset["X"]
    X_sp = dataset["X_sp"].numpy()
    to_idx = np.vectorize(lambda x: strain_ids.index(x))
    X_sp = to_idx(X_sp)

    model = SPgMLP(
        dim = 512,
        depth = 32,
        seq_len = 700,
        num_tokens = 66,
        ff_mult = 2,
        heads = 1,
        circulant_matrix = True,
        act = nn.Tanh(),
        prob_survival = 1.0,
        n_species = 34
    ).cuda()
    
    pt = torch.load("./multisp.pt", lambda storage, loc: storage.cuda(0))
    model.load_state_dict(pt["model"])
    del pt

    with torch.no_grad():
        X_e = torch.empty(0, 700, 512).cuda()
        for x, x_sp in tqdm(zip(torch.split(X, 1000), torch.split(torch.tensor(X_sp).unsqueeze(1), 1000))):
            out = model.get_output(x.cuda(non_blocking=True), x_sp.cuda(non_blocking=True))
            X_e = torch.cat((X_e, out), dim=0)

    dataset["X_e"] = X_e.cpu()

    torch.save(dataset, path)

def load_dataset(path, strain_ids, test_pairs=(), mode=MODE_SP_ID):
    dataset = torch.load(path)
    X = dataset["X"]
    Y = dataset["Y"]
    #X_e = dataset["X_e"]
    X_id = dataset["X_id"].numpy()
    Y_id = dataset["Y_id"].numpy()
    X_sp = dataset["X_sp"].numpy()
    Y_sp = dataset["Y_sp"].numpy()

    test_idx = np.logical_and(X_sp==test_pairs[0], Y_sp==test_pairs[1])
    if mode == MODE_SP_ID:
        train_idx = ~test_idx
    elif mode == MODE_SP_OD:
        train_idx = Y_sp!=test_pairs[1]

    to_idx = np.vectorize(lambda x: strain_ids.index(x))
    X_sp = to_idx(X_sp)
    Y_sp = to_idx(Y_sp)

    #X_e_train, X_e_test = X_e[train_idx], X_e[test_idx]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    X_id_train, X_id_test = X_id[train_idx], X_id[test_idx]
    Y_id_train, Y_id_test = Y_id[train_idx], Y_id[test_idx]
    X_sp_train, X_sp_test = X_sp[train_idx], X_sp[test_idx]
    Y_sp_train, Y_sp_test = Y_sp[train_idx], Y_sp[test_idx]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    X_id_train, X_id_test = X_id[train_idx], X_id[test_idx]
    Y_id_train, Y_id_test = Y_id[train_idx], Y_id[test_idx]
    X_sp_train, X_sp_test = X_sp[train_idx], X_sp[test_idx]
    Y_sp_train, Y_sp_test = Y_sp[train_idx], Y_sp[test_idx]

    X_e_train, X_e_test = torch.randn(X_train.size(0), X_train.size(1), 512), torch.randn(X_test.size(0), X_test.size(1), 512)

    dataset_train = {
        "X": X_train,
        "Y": Y_train,
        "X_e": torch.zeros(X_train.size(0), 700, 512),
        "X_id": X_id_train,
        "Y_id": Y_id_train,
        "X_sp": X_sp_train,
        "Y_sp": Y_sp_train
    }

    dataset_test = {
        "X": X_test,
        "Y": Y_test,
        "X_e": torch.zeros(X_test.size(0), 700, 512),
        "X_id": X_id_test,
        "Y_id": Y_id_test,
        "X_sp": X_sp_test,
        "Y_sp": Y_sp_test
    }

    # calculate sample weight by inverse number of samples
    #w = get_sample_weights(Y_id)
    #w_train, w_test = w[train_idx], w[test_idx]

    dataset_train = TensorDataset(X_e_train, X_train, Y_train, torch.tensor(Y_sp_train).unsqueeze(1))
    dataset_test = TensorDataset(X_e_test, X_test, Y_test, torch.tensor(Y_sp_test).unsqueeze(1))

    return dataset_train, dataset_test

def many_to_many_(Model, config, dataset_train, dataset_test, n_epochs, batch_size, log_interval=100, parent_dir="./Result", weight_path="./Result/weight", device_ids=[0,1]):
    # loader
    loader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    # model
    model = Model(**config).cuda()
    model = nn.DataParallel(model, device_ids=device_ids)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4*len(device_ids))
    criterion = nn.CrossEntropyLoss()

    for i in tqdm(range(1, n_epochs + 1)):
        for j, (x_e, x, y, y_sp) in enumerate(loader_train):
            optimizer.zero_grad()
            out = model(x_e.cuda(non_blocking=True), x.cuda(non_blocking=True), y_sp.cuda(non_blocking=True))
            
            loss = criterion(out.permute(0,2,1), y.cuda(non_blocking=True))
            loss.backward()
            
            optimizer.step()

        if i==1 or i%log_interval==0:
            # 訓練データでの変換
            write_fna_faa(x, "{}/train/src_{}".format(parent_dir, i), "SRC")
            write_fna_faa(y, "{}/train/tgt_{}".format(parent_dir, i), "TGT")
            write_fna_faa(torch.argmax(out, dim=2).cpu(), "{}/train/gen_{}".format(parent_dir, i), "GEN")

            # 検証データでの変換
            with torch.no_grad():
                for j, (x_e, x, y, y_sp) in enumerate(loader_test):
                    out = model(x_e.cuda(non_blocking=True), x.cuda(non_blocking=True), y_sp.cuda(non_blocking=True))

                    write_fna_faa(x, "{}/test/src_{}".format(parent_dir, i), "SRC")
                    write_fna_faa(y, "{}/test/tgt_{}".format(parent_dir, i), "TGT")
                    write_fna_faa(torch.argmax(out, dim=2).cpu(), "{}/test/gen_{}".format(parent_dir, i), "GEN")

                    break

    torch.save(model.module.state_dict(), weight_path)

def distributed_data_parallel(gpu, world_size, Model, config, dataset_path, strain_ids, test_pairs, n_epochs, batch_size, log_interval=100, parent_dir="./Result", use_apex=True):
    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    # define model
    model = Model(**config).to(gpu)
    optimizer = optim.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss()

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    ddp_model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu])

    dist.barrier()

    dataset_train, dataset_test = load_dataset(rank, dataset_path, strain_ids, test_pairs)

    # loader
    sampler = DistributedSampler(
        dataset_train,
        num_replicas=world_size,
        rank=rank
    )
    loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
    )
    loader_test = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    # scheduler
    scheduler = OneCycleLR(
        optimizer,
        lr=1e-4,
        total_steps=(n_epochs*len(loader))//world_size,
        pct_start=0.1,
        anneal_strategy='linear'
    )

    # training
    for i in tqdm(range(1, n_epochs + 1)):
        sampler.set_epoch(i)
        for j, (x_e, x, y, y_sp) in enumerate(loader):
            optimizer.zero_grad()
            out = ddp_model(x_e.cuda(non_blocking=True), x.cuda(non_blocking=True), y_sp.cuda(non_blocking=True))
            
            loss = criterion(out.permute(0,2,1), y.cuda(non_blocking=True))
            loss.backward()
            
            optimizer.step()

            scheduler.step()

        if (i==1 or i%log_interval==0) and rank == 0:
            # 訓練データでの変換
            write_fna_faa(x, "{}/train/src_{}".format(parent_dir, i), "SRC")
            write_fna_faa(y, "{}/train/tgt_{}".format(parent_dir, i), "TGT")
            write_fna_faa(torch.argmax(out, dim=2).cpu(), "{}/train/gen_{}".format(parent_dir, i), "GEN")

            # 検証データでの変換
            with torch.no_grad():
                for j, (x_e, x, y, y_sp) in enumerate(loader_test):
                    out = model(x_e.cuda(non_blocking=True), x.cuda(non_blocking=True), y_sp.cuda(non_blocking=True))

                    write_fna_faa(x, "{}/test/src_{}".format(parent_dir, i), "SRC")
                    write_fna_faa(y, "{}/test/tgt_{}".format(parent_dir, i), "TGT")
                    write_fna_faa(torch.argmax(out, dim=2).cpu(), "{}/test/gen_{}".format(parent_dir, i), "GEN")

                    break

    dist.destroy_process_group()