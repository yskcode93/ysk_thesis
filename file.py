import re
import itertools
import torch
import numpy as np
# biopython
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
# fetch all
from database import fetchall
#from .analysis import translation_map


MAX_CODON_LENGTH = 700 # 入力するコドン列の最大長
MAX_LEN_DIFF = 0.03 # parsers.pyで使用する。

# This is the standard residue order when coding codon type as a number.
CODONS = itertools.product("ACGT", repeat=3) # all types of condon
RESTYPES = ["".join(c) for c in CODONS]
RESTYPE_ORDER = {restype: i for i, restype in enumerate(RESTYPES)}
RESTYPE_NUM = len(RESTYPES)  # := 64.
RESTYPE_PAD_NUM = RESTYPE_NUM + 1 # := 65
RESTYPES_GAP = RESTYPES + ['---']
RESTYPES_GAP_ORDER = {restype: i for i, restype in enumerate(RESTYPES_GAP)}

PAD_RESTYPES = [''] + RESTYPES
PAD_RESTYPES_GAP = [''] + RESTYPES + ['']
TGT_FAA_PATH = '/home/slab/wataru_hiratani/work/data/sonicparanoid_exec/sonic_input/GCF_000009045.1_ASM904v1_translated_cds.faa' # B.subtillisのaas配列
TGT_FNA_PATH = '/home/slab/wataru_hiratani/work/data/sonicparanoid_exec/cds_fna/GCF_000009045.1_ASM904v1_cds_from_genomic.fna' # B.subtillisのnucs配列

USED_SPECIES_PATH = "/home/slab/wataru_hiratani/work/used_species.csv"


#いつものやつ
#codons = itertools.product("ATGC", repeat=3)
#codons = ["".join(c) for c in codons]
#vocab = ["<pad>"] + codons + ["<msk>"]

 #平谷のやつ1/31
#進捗用
codons = itertools.product("ACGT", repeat=3)
codons = ["".join(c) for c in codons]
vocab = ["<pad>"] + codons + ["<msk>"]


translation_map = {
    "TTT": "F", "TTC": "F",
    "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y",
    "CAT": "H", "CAC": "H",
    "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D",
    "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
    "ATG": "M",
    "TGG": "W",
    
    "TAA":'*','TAG':'*','TGA':'*'
}

AAS = sorted(''.join(list(set(translation_map.values()))))
PAD_AAS_GAP = [''] + AAS + ['']

to_AAS_ORDER = {PAD_RESTYPES_GAP.index(x) : PAD_AAS_GAP.index(y) for x,y in translation_map.items()}
to_PAD_AAS_GAP_INDEX = [-100 for x in range(len(PAD_RESTYPES_GAP))]
for k,v in to_AAS_ORDER.items():
    to_PAD_AAS_GAP_INDEX[k] = v
to_PAD_AAS_GAP_INDEX = np.array(to_PAD_AAS_GAP_INDEX)
to_PAD_AAS_GAP_INDEX[0] = 0 # PAD
to_PAD_AAS_GAP_INDEX[65] = 22 # GAP

# 4 BASE of the nucleotide.
NUCS = ['A', 'C', 'G', 'T']

def to_PAD_BASE_GAP_INDEX(position:int):

    to_base_order = {PAD_RESTYPES_GAP.index(x) : NUCS.index(x[position - 1]) for x in translation_map.keys()}
    to_pad_base_gap_index = [-100 for x in range(len(PAD_RESTYPES_GAP))]
    for k,v in to_base_order.items():
        to_pad_base_gap_index[k] = v
    to_pad_base_gap_index = np.array(to_pad_base_gap_index)
    to_pad_base_gap_index[0] = 0 # PAD
    to_pad_base_gap_index[65] = 5 # GAP

    return to_pad_base_gap_index

to_PAD_FB_GAP_INDEX = to_PAD_BASE_GAP_INDEX(1)
to_PAD_SB_GAP_INDEX = to_PAD_BASE_GAP_INDEX(2)
to_PAD_tB_GAP_INDEX = to_PAD_BASE_GAP_INDEX(3)

class GeneBatchLoader():
    def __init__(self, sql, batch_size, dataset_size):
        self.sql = sql
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.last_id = 0

        self._i = 0

    def __iter__(self):
        return self
    
    def __len__(self):
        return self.dataset_size // self.batch_size if self.dataset_size % self.batch_size == 0 else self.dataset_size // self.batch_size

    def __next__(self):
        if self._i >= self.__len__():
            self._i = 0
            raise StopIteration()

        params = (self.last_id, self.batch_size)
        self._i += 1

        return fetchall(self.sql, params)

### function sanitize replaces ambiguous symbols in a sequence with "A", "T", "G" and "C".
## Arguments
# seq: genome sequence in String
## Return Values
# seq: genome sequence in String without ambiguous symbols
def sanitize(seq):
    seq = re.sub(r"R", ["A", "G"][np.random.randint(2)], seq)
    seq = re.sub(r"Y", ["T", "C"][np.random.randint(2)], seq)
    seq = re.sub(r"K", ["G", "T"][np.random.randint(2)], seq)
    seq = re.sub(r"S", ["G", "C"][np.random.randint(2)], seq)
    seq = re.sub(r"W", ["A", "T"][np.random.randint(2)], seq)
    seq = re.sub(r"B", ["T", "G", "C"][np.random.randint(3)], seq)
    seq = re.sub(r"D", ["A", "G", "T"][np.random.randint(3)], seq)
    seq = re.sub(r"H", ["A", "C", "T"][np.random.randint(3)], seq)
    seq = re.sub(r"M", ["A", "C"][np.random.randint(2)], seq)
    seq = re.sub(r"V", ["A", "C", "G"][np.random.randint(3)], seq)
    seq = re.sub(r"N", ["A", "C", "T", "G"][np.random.randint(4)], seq)

    return seq

# convert codon to amino acids
def to_aa(codon):
    return "".join(map(lambda x: translation_map[x] if x in translation_map else "", re.split('(...)', codon)[1:-1:2]))

# convert to strings
def to_seq(x, vocab):
    return np.apply_along_axis(lambda x: ''.join(x), 0, np.array(vocab)[x].T)

# save sequences to file
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

def write_fna_faa(x, basename, prefix):
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

    # pad out after termination
    for ter in [17, 19, 25]:
        mask = torch.cumsum(x == ter, 1)
        mask[x==ter] = 0
        x[mask.bool()] = 0

    x_str = to_seq(x.cpu(), vocab)
    x_str = [x.translate(trans) for x in x_str]
    x_aa = [to_aa(x) for x in x_str]

    to_file(x_str, "{}.fna".format(basename), "", prefix)
    to_file(x_aa, "{}.faa".format(basename), "", prefix)

def write_fna_faa_2(x, basename, prefix):
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

    # pad out after termination
    for ter in [49, 51, 57]:
        mask = torch.cumsum(x == ter, 1)
        mask[x==ter] = 0
        x[mask.bool()] = 0

    x_str = to_seq(x.cpu(), vocab)
    x_str = [x.translate(trans) for x in x_str]
    x_aa = [to_aa(x) for x in x_str]

    to_file(x_str, "{}.fna".format(basename), "", prefix)
    to_file(x_aa, "{}.faa".format(basename), "", prefix)
    
