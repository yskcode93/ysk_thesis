# pytorch
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
import time
import numpy as np
from tqdm import tqdm
from Bio import SeqIO

# inference mode
MODE_GREEDY = 0
MODE_BEAM = 1

def test_from_fasta(ConverterClass, config, strain_ids, sp, n_epochs, n_folds, mode, device,\
    pretrain, PretrainClass, config_pretrain, pretrain_path_en, pretrain_path_de,\
    output_dir="./Result", batch_size=100):
    """This function generates sequences from source sequences in "{output_dir}/finetune" by greedy search or beam search

    Argument:
    ConverterClass -- Class: custom model class that extends torch.nn.Module
    config -- Dictionary: arguments of ConverterClass
    strain_ids -- List(Integer): list of internal ids for strains.
    sp -- Integer: internal id of the target strain(Y)
    n_epochs -- Integer: the model weights saved in "checkpoint_{}_{n_epochs}" are used for inference.
    n_folds -- Integer: the number of folds in cross validation
    mode -- Integer{0,1}: inference mode, 0 for greedy search, 1 for beam search
    device -- Integer: which gpu is used for inference
    pretrain -- Boolean: whether or not output feature vectors of the pretrained model are used in training.
    PretrainClass -- Class: custom model class that extends torch.nn.Module
    config_pretrain -- Dictionary: arguments of PretrainClass
    pretrain_path -- String: Path in which the model weights are saved
    output_dir -- String: directory for saving results. Generated sequences are saved in "{output_dir}/finetune".
    batch_size -- Integer: batch size for beam search Set smaller value when memory is overflowed.

    Return:
    None
    """

    for i in tqdm(range(1, n_folds+1)):
        # source sequences
        with open("{}/finetune/src_{}.fna".format(output_dir, i), "r") as f:
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
        pt = torch.load("{}/weight/checkpoint_{}_{}.pt".format(output_dir, i, n_epochs), map_location=lambda storage, loc: storage.cuda(device))
        model = ConverterClass(**config).to(device)
        model.load_state_dict(pt["model"])
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
            else:
                X_e = X

            if mode == MODE_GREEDY:
                out = model.infer_greedily(X_e, X, Y_sp, X.size(1), [17, 19, 25])
            elif mode == MODE_BEAM:
                outs = []
                for x_e, x, y_sp in zip(torch.split(X_e, batch_size), torch.split(X, batch_size), torch.split(Y_sp, batch_size)):
                    #out = model.beam_search(x_e, x, y_sp, x.size(1), 5, [17, 19, 25])
                    #試しにbeam_search使わずに                    
                    out = model(x_e, x, y_sp, x.size(1), 5, [17, 19, 25])
                    if out.size(1) < 700:
                        out = torch.cat((out.cpu(), torch.zeros(out.size(0), 700-out.size(1)).long()), dim=1)
                    outs.append(out.cpu())         
                
                out = torch.cat(outs)
                
            if pretrain:
                out = model_pretrain_de.get_output_decoder(torch.cat((cls, out), dim=1)) if "n_species" in config_pretrain else model_pretrain_de.get_output_decoder(out)  
             

        write_fna_faa(out.cpu(), "{}/finetune/gen_{}".format(output_dir, i), "GEN")        
