import os
from Bio import SeqIO, pairwise2

import re
import string
import itertools
import math
import subprocess
import time
from tqdm import tqdm

from .hmmer import sequence_search, check, retrieve_result

# translation mapping
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
    "TGG": "W"
}

# calculate GC Content
def gc_content(seq):
    return  (seq.count("G") + seq.count("C")) / len(seq)

# calculate RSCU(Relative Synonymous Codon Usage)
def rscu(reference):
    codon_map = {k: 0 for k in translation_map.keys()}
    aa_max = {v: 0 for v in translation_map.values()}

    with open(reference, "r") as f:
        ref = SeqIO.parse(f, "fasta")
        for rec in ref:
            seq = str(rec.seq)
            for i in range(0, len(seq), 3):
                codon = seq[i:i+3]
                if codon in codon_map:
                    codon_map[codon] += 1
                    
    for key in codon_map.keys():
        aa = translation_map[key]
        if codon_map[key] > aa_max[aa]:
            aa_max[aa] = codon_map[key]
        
    return {k: codon_map[k] / aa_max[v] for k, v in translation_map.items()}

def log_cai(seq, rscu):
    log_cai = 0
    for codon in re.split('(...)',seq)[1:-1:2]:
        if codon in rscu:
            log_cai += math.log(rscu[codon])
            
    return log_cai / (len(seq) // 3)

from dataclasses import dataclass

@dataclass
class Alignment:
    seqA: str
    seqB: str
    score: float

LEVEL_AMINO = 0
LEVEL_CODON = 1
LEVEL_NUCLEOTIDE = 2

os.environ["PATH"] += ":/home/jupyter-user/seq-align/bin"
def alignment(x, y, level):
    if level == LEVEL_AMINO:
        # blastp configuration
        proc = subprocess.run(["needleman_wunsch", "--scoring", "BLOSUM62", "--printscores", x, y], \
                  stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        res = proc.stdout.decode("utf8")
        seqA, seqB, score = res.split("\n")[:3]
        score = float(score[6:])
        a = Alignment(seqA, seqB, score)

    elif level == LEVEL_CODON:
        codons = itertools.product("ATGC", repeat=3)
        codons = ["".join(c) for c in codons]
        base64 = string.ascii_letters + string.digits + "+/"
        x = "".join(map(lambda x: base64[codons.index(x)] if x in codons else "*", re.split('(...)',x)[1:-1:2]))
        y = "".join(map(lambda x: base64[codons.index(x)] if x in codons else "*", re.split('(...)',y)[1:-1:2]))
        [a] = pairwise2.align.globalms(x, y, 1.9, 0, -10, -0.1, one_alignment_only=True)

    elif level == LEVEL_NUCLEOTIDE:
        proc = subprocess.run(["needleman_wunsch", "--printscores", x, y], \
                  stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        res = proc.stdout.decode("utf8")
        seqA, seqB, score = res.split("\n")[:3]
        score = float(score[6:])
        a = Alignment(seqA, seqB, score)

    return a

def identity(alignment):
    match = 0
    for charA, charB in zip(alignment.seqA, alignment.seqB):
        match += charA == charB

    return match / len(alignment.seqA)

def nc_similarity(x, y):
    match = 0
    n_gaps = x.count("-") + y.count("-")
    for char_x, char_y in zip(x, y):
        match += char_x == char_y

    return match / (len(x) - n_gaps) if len(x) - n_gaps != 0 else 0

def amino_to_codon(x, y, alignment):
    for i in range(0, len(alignment.seqA)):
        if alignment.seqA[i] == "-":
            x = x[:i*3] + "---" + x[i*3:]
            continue
        if alignment.seqB[i] == "-":
            y = y[:i*3] + "---" + y[i*3:]
            continue

    return x, y

def optimize_codon(seq, rscu):
    seq = re.split('(...)', seq)[1:-1:2]
    optimized = ""

    optimization_dict = {}
    for key, value in rscu.items():
        if value < 1: continue

        aa = translation_map[key]
        optimization_dict[aa] = key

    for codon in seq:
        if codon not in translation_map:
            optimized += codon
        elif rscu[codon] < 1:
            optimized += optimization_dict[translation_map[codon]]
        else:
            optimized += codon

    return optimized
    
def domain_search(seq):
    result = []
    for s in tqdm(seq):
        query = ">query\n"
        query += str(s.seq)

        task_id = sequence_search(query)
        
        while not check(task_id):
            time.sleep(2)

        res = retrieve_result(task_id)

        if res is None:
            result.append(res)
            continue
        else:
            cath_ids = []
            for hit in res["funfam_scan"]["results"][0]["hits"]:
                cath_ids.append(hit["match_cath_id"]["id"])
            cath_ids = set(cath_ids)
            result.append(cath_ids)

    return result