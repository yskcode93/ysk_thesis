#fastasはfileのパスが入ったリスト
#SepIO.parseで指定したパスのfastaファイルを読み込める

import torch
from torch.nn.utils.rnn import pad_sequence

import re
from itertools import product
from Bio import Seq, SeqIO
from onehot import one_hot, featurize
import numpy as np

codons = product("ATGC", repeat=3)
codons = ["".join(c) for c in codons]
vocab = ["<pad>"] + codons + ["<msk>", "<cls>"] 
#vocab = ["<pad>"] + codons + ["<msk>"] 


#最後にpadを右に持っていくやつ
def right_shift_zeros(arr):
    # マスクを作成して、各要素がゼロかどうかを判定
    mask = arr != 0
    # ゼロでない要素の数をカウント
    counts = mask.sum(axis=1)
    # 新しい配列を作成し、ゼロで初期化
    new_arr = np.zeros_like(arr)
    # 各行に対してゼロでない要素を左に詰める
    for i in range(len(arr)):
        new_arr[i, :counts[i]] = arr[i, mask[i]]
    return new_arr

def fasta_to_data_eos(fastas: list, slen= 700):
    X = []
    for fasta in fastas:
        for record in SeqIO.parse(fasta, 'fasta'):
            seq = record.seq
            if any(char not in 'ATGC' for char in seq):
                continue
            if len(seq) // 3 <= slen:
                seq = re.split('(...)', str(seq))[1:-1:2]
                seq = list(map(lambda x: vocab.index(x), seq))
                seq = torch.tensor(seq)
                tmp = torch.tensor([66])
                seq2 = torch.cat([seq, tmp], dim=-1)
                X.append(seq2)
            else:
                path_w = './test_wsss.txt'
                with open(path_w, mode='a') as f:
                    f.write("seq\n")
                    f.write(str(seq))
                    
    base = torch.zeros(701)
    X.append(base)
    X = pad_sequence(X, batch_first= True)
    X = X.to('cpu').detach().numpy().copy()
    X = np.delete(X, X.shape[0]-1, 0)

    return X

def fasta_to_data(fastas: list, slen= 700):
    # return X: B x T x *
    # return y: B * Ny

    X = []
    #y = []
    for fasta in fastas:
        for record in SeqIO.parse(fasta, 'fasta'):
            seq = record.seq
            if any(char not in 'ATGC' for char in seq):
                continue
            if len(seq) // 3 <= slen:
                seq = re.split('(...)', str(seq))[1:-1:2]
                seq = list(map(lambda x: vocab.index(x), seq))
                #seq = torch.tensor([vocab.index("<cls>")] + seq)
                seq = torch.tensor(seq)
                
                X.append(seq)
                #y.append(ids.index(ID))

    #print(y)
    base = torch.zeros(700)
    X.append(base)
    X = pad_sequence(X, batch_first= True)
    #y = torch.tensor(y)

    return X

def fasta_to_data_(fastas: list, slen= 700):
    # return X: B x T x *
    # return y: B * Ny

    X = []
    #y = []
    for fasta in fastas:
        for record in SeqIO.parse(fasta, 'fasta'):
            seq = record.seq
            if len(seq) // 3 <= slen:
                X.append(seq)
    return X

def fasta_to_data_2(fastas: list, slen= 700):
    # return X: B x T x *
    # return y: B * Ny

    X = []
    #y = []
    for fasta in fastas:
        for record in SeqIO.parse(fasta, 'fasta'):
            seq = record.seq
            if len(seq) // 3 <= slen:
                X.append(seq)
    return X

def fasta_to_data_onehot(fastas: list, slen= 700):
    # return X: B x T x *
    # return y: B * Ny
    cnt = 0
    #y = []
    for fasta in fastas:
        for record in SeqIO.parse(fasta, 'fasta'):
            seq = record.seq
            if len(seq) // 3 <= slen:
                tmp = featurize(seq)
                if(cnt==0):
                    res = tmp
                    cnt = cnt + 1
                else:
                    res = np.vstack((res, tmp))
                    
    return res

def fasta_to_data_aa(fastas: list, slen= 700):
    # return X: B x T x *
    # return y: B * Ny

    path_w3 = './test_w3.txt'
    X = []
    #y = []
    for fasta in fastas:
        for record in SeqIO.parse(fasta, 'fasta'):
            seq_aa = []
            seq = record.seq
            with open(path_w3, mode='a') as f:
                    f.write("seq\n")
                    f.write(str(seq))
                    f.write("\n")
                    f.write(str(len(seq)))
                    f.write("\n")     
            if len(seq) <= slen: 
                for i in seq:
                    if(i == "G"):
                        seq_aa.append(1)
                    elif(i == "A"):
                        seq_aa.append(2)
                    elif(i == "V"):
                        seq_aa.append(3)
                    elif(i == "L"):
                        seq_aa.append(4)
                    elif(i == "I"):
                        seq_aa.append(5)
                    elif(i == "P"):
                        seq_aa.append(6)
                    elif(i == "M"):
                        seq_aa.append(7)                        
                    elif(i == "F"):
                        seq_aa.append(8)
                    elif(i == "W"):
                        seq_aa.append(9)
                    elif(i == "S"):
                        seq_aa.append(10)
                    elif(i == "T"):
                        seq_aa.append(11)  
                    elif(i == "N"):
                        seq_aa.append(12)
                    elif(i == "Q"):
                        seq_aa.append(13)
                    elif(i == "Y"):
                        seq_aa.append(14)
                    elif(i == "C"):
                        seq_aa.append(15)
                    elif(i == "D"):
                        seq_aa.append(16)
                    elif(i == "E"):
                        seq_aa.append(17)                        
                    elif(i == "H"):
                        seq_aa.append(18)
                    elif(i == "K"):
                        seq_aa.append(19)
                    elif(i == "R"):
                        seq_aa.append(20)
                #seq = list(map(lambda x: vocab.index(x), seq))
                seq = torch.tensor([vocab.index("<cls>")] + seq_aa)
                #seq = torch.tensor(seq)
                X.append(seq)
                #y.append(ids.index(ID))
        
    with open(path_w3, mode='a') as f:
        f.write(str(X))
        f.write("\n")
        f.write(str(len(X)))
        f.write("\n")        
        
    #print(y)
    X = pad_sequence(X, batch_first= True)
    #y = torch.tensor(y)

    return X

def fasta_to_data2(tfastas: list, sfastas: list, slen= 700):
    # return X: B x T x *
    # return y: B * Ny

    X_t = []
    y_t = []
    X_s = []
    y_s = []
    tc = 0
    sc = 0
    delist = []
    for fasta in tfastas:
        #書き換え部分↓
        #ID = fasta.split('_')[1][:-2]
        ID = '002119445'
        for record in SeqIO.parse(fasta, 'fasta'):
            seq = record.seq
            if len(seq) // 3 <= slen:
                seq = re.split('(...)', str(seq))[1:-1:2]
                seq = list(map(lambda x: vocab.index(x), seq))
                seq = torch.tensor([vocab.index("<cls>")] + seq)
                #seq = torch.tensor(seq)      
            else:
                seq = re.split('(...)', str(seq))[1:-1:2]
                seq = list(map(lambda x: vocab.index(x), seq))
                seq = torch.tensor([vocab.index("<cls>")] + seq)
                delist.append(tc)
                
            
            X_t.append(seq)
            y_t.append(tc) 
            tc+=1  
   
    for fasta in sfastas:
        #書き換え部分↓
        #ID = fasta.split('_')[1][:-2]
        ID = '000009045'
        for record in SeqIO.parse(fasta, 'fasta'):
            seq = record.seq
            if len(seq) // 3 <= slen:
                seq = re.split('(...)', str(seq))[1:-1:2]
                seq = list(map(lambda x: vocab.index(x), seq))
                seq = torch.tensor([vocab.index("<cls>")] + seq)
                #seq = torch.tensor(seq)
            else:
                seq = re.split('(...)', str(seq))[1:-1:2]
                seq = list(map(lambda x: vocab.index(x), seq))
                seq = torch.tensor([vocab.index("<cls>")] + seq)
                delist.append(sc)
                
            X_s.append(seq)
            y_s.append(sc)      
            sc+=1
            
    #X = pad_sequence(X, batch_first= True)
    #y = torch.tensor(y)
    
    delist = list(set(delist))
    #print(len(delist))
    #print(len(X_t))
    #print(len(y_t))
    
    for i in range(len(X_t)):
        if i in delist:
            X_t[i] = -1
            
    for i in range(len(y_t)):
        if i in delist:
            y_t[i] = -1    
            
    for i in range(len(X_s)):
        if i in delist:
            X_s[i] = -1   
            
    for i in range(len(y_s)):
        if i in delist:
            y_s[i] = -1          
    
    target = -1
    X_t = [item for item in X_t if type(item) != int]
    y_t = [item for item in y_t if type(item) != int]
    X_s = [item for item in X_s if type(item) != int]
    y_s = [item for item in y_s if type(item) != int] 

    for i in range(len(X_t)):
        y_t.append(0)
        y_s.append(0)
        
    print(y_t)    

    X_t = pad_sequence(X_t, batch_first= True)
    y_t = torch.tensor(y_t)
    X_s = pad_sequence(X_s, batch_first= True)
    y_s = torch.tensor(y_s)
    
    return X_t, y_t, X_s, y_s
