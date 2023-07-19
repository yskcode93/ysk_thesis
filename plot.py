#!/usr/bin/env python
# coding: utf-8

# In[1]:


path = "Result/pretrain/protein_family_subtilis.log"

MLMloss=[]
CLSloss=[]

with open(path) as f:
    for line in f:
        MLMloss.append(float(line[22:28]))
        CLSloss.append(float(line[34:40]))
print(MLMloss)
print(CLSloss)


# In[ ]:


import os
import urllib.request
import shutil
import gzip

download_folder_a ="./Data/ideonella/amino"  # ダウンロード先フォルダのパスを指定
download_folder_c ="./Data/ideonella/codon"  # ダウンロード先フォルダのパスを指定
df = pd.read_csv("./Data/ideonella/ideonella.csv")

for url in df['RefSeq FTP']:
    if(type(url) == float):
        continue
        
    o = urlparse(url)
    
    path_faa = o.path + "/" + os.path.basename(url) + '_translated_cds.faa.gz'
    url_faa = urlunparse(o._replace(scheme='https',path=path_faa))
    path_fna = o.path + "/" + os.path.basename(url) + '_cds_from_genomic.fna.gz'
    url_fna = urlunparse(o._replace(scheme='https',path=path_fna))

    if os.path.exists(os.path.basename(os.path.basename(url) + '_cds_from_genomic.fna.gz')):
         continue
    if os.path.exists(os.path.join(download_folder_a, os.path.basename(url) + '_translated_cds.faa')):
        continue
    
    print("downloading", url_fna)
    print("downloading", url_faa)
    
    file_path_a = os.path.join(download_folder_a, os.path.basename(url) + '_translated_cds.faa.gz')
    file_path_c = os.path.join(download_folder_c, os.path.basename(url) + '_cds_from_genomic.fna.gz')
    
    urllib.request.urlretrieve(url_faa, file_path_a)
    with gzip.open(file_path_a, 'rb') as f_in, open(os.path.join(download_folder_a, os.path.basename(url) + '_translated_cds.faa'), 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(file_path_a)
    
    urllib.request.urlretrieve(url_fna, file_path_c)
    with gzip.open(file_path_c, 'rb') as f_in, open(os.path.join(download_folder_c, os.path.basename(url) + '_cds_from_genomic.fna'), 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(file_path_c)
    
    #urllib.request.urlretrieve(url_faa, file_path)
    #with gzip.open(file_path, 'rb') as f_in, open(os.path.join(download_folder, os.path.basename(url) + '_translated_cds.faa'), 'wb') as f_out:
    #    shutil.copyfileobj(f_in, f_out)
    #os.remove(file_path)


# In[1]:


import torch
import torch.multiprocessing as mp
from convert import train, cross_validate, cross_validate_, cross_validate_stratified, MODE_GREEDY, MODE_BEAM
from model import Converter, TransformerConverter, FastAutoregressiveConverter
from model2 import MultigMLP
import torch.nn as nn


# In[7]:


output_dir = "./Result/pretrain/protein_family_stratified"
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
i = 1
n_epochs = 2000
device  = 0
ConverterClass = FastAutoregressiveConverter
config = config_transformer




pt = torch.load("{}/weight/checkpoint_{}_{}.pt".format(output_dir, i, n_epochs), map_location=lambda storage, loc: storage.cuda(device))
model = ConverterClass(**config).to(device)
        #print(model)
model.load_state_dict(pt["model"],strict=False)
model.eval()
del pt

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


pt = torch.load(pretrain_path_en, map_location=lambda storage, loc: storage.cuda(device))
model_pretrain_en = PretrainClass(**config_pretrain).to(device)
model_pretrain_en.load_state_dict(pt["model"])
model_pretrain_en.eval()
del pt                  
            
pt = torch.load(pretrain_path_de, map_location=lambda storage, loc: storage.cuda(device))
model_pretrain_de = PretrainClass(**config_pretrain).to(device)
model_pretrain_de.load_state_dict(pt["model"])
model_pretrain_de.eval()
del pt

cls = torch.tensor([[len(vocab)]]*X.size(0)).to(device)
X_e = model_pretrain_en.get_output(torch.cat((cls, X), dim=1)) if "n_species" in config_pretrain else model_pretrain_en.get_output(X)

for x_e, x, y_sp in zip(torch.split(X_e, batch_size), torch.split(X, batch_size), torch.split(Y_sp, batch_size)):
    out = model(x_e, x, y_sp)
    if pretrain:
        out = model_pretrain_de.get_output_decoder(out)
        
        


# In[1]:


# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
# basic
import math

from random import randrange
# gated MLP
from g_mlp_pytorch import gMLP
# transformer
from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder, RecurrentEncoderBuilder, RecurrentDecoderBuilder
from fast_transformers.masking import LengthMask, TriangularCausalMask

from file import to_seq, vocab
from model2 import MultigMLP

premodel = MultigMLP(67,512,1024,701,32,nn.Tanh(),278,)
pt = torch.load("./Result/pretrain/protein_family_subtilis.pt")
        #self.decoder = self.premodel.to(device)
decoder = premodel.load_state_dict(pt["model"])
decoder


# In[7]:


path = "Result/pretrain/protein_family_stratified/finetune.log"
loss = []
with open(path) as f:
    for line in f:
        loss.append(float(line[25:31]))
print(loss[19:39])

import matplotlib.pyplot as plt
x1 = loss[0:20]
x2 = loss[20:40]
x3 = loss[40:60]
x4 = loss[60:80]
x5 = loss[80:100]
x6 = loss[100:120]
#plt.plot(x1,color='#ff7f00');
plt.plot(x2,color='#ff7f00');
plt.plot(x3,color='#ff7f00');
plt.plot(x4,color='#ff7f00');
plt.plot(x5,color='#ff7f00');
plt.plot(x6,color='#ff7f00');


# In[14]:


import matplotlib.pyplot as plt
path = "Result/pretrain/protein_family_stratified/for_7_11.log"
loss = []
i  = 0
with open(path) as f:
    for line in f:
        i =  i + 1
        if(i >= 1 and i<=60 ):
            loss.append(float(line[18:25]))
print(loss)

x1 = loss

plt.plot(x1,color='#ff7f00');
#plt.savefig("./Result/pretrain/protein_family_stratified/convert_png/M2_4_10/pretrain_1_24lay/loss.png")


# In[1]:


path = "CDSdata/thuringiensis/fna.fna"
cnt = 0
with open(path) as f:
    for line in f:
        if('>' in line):
            cnt = cnt + 1
print(cnt)


# In[2]:


path = "CDSdata/subtilis/fna.fna"
cnt = 0
with open(path) as f:
    for line in f:
        if('>' in line):
            cnt = cnt + 1
print(cnt)


# In[22]:


import torch
import torch.nn as nn
import numpy as np
subpt = torch.load("../thesis_code/Result/pretrain/protein_family_subtilis.pt")
subweight = subpt['model']['to_logits.1.weight']
decoder = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512,67)
        )
#decoder = nn.Parameter(subweight)
print(decoder)
sa  = np.zeros([67,512])
x = torch.from_numpy(sa.astype(np.float32)).clone()
a = decoder(x)
a


# In[2]:


import matplotlib.pyplot as plt
x = MLMloss
plt.plot(x,color='#ff7f00');


# In[3]:


import matplotlib.pyplot as plt
x = CLSloss
plt.plot(x);


# In[46]:


import torch
model = torch.load("../thesis_code/Result/pretrain/protein_family_thuringiensis.pt")


# model

# In[51]:


model.keys()


# In[53]:


model['model'].keys()


# In[61]:


weight = model['model'][ 'to_logits.1.weight'] #decoderの重み
weight.shape


# In[65]:


weight


# In[62]:


import torch
import torch.nn as nn
decoder = nn.Sequential(
            nn.LayerNorm(67),
            nn.Linear(67, 512)
        )


# In[63]:


decoder = weight


# In[64]:


decoder


# In[70]:



import numpy as np
import glob
import re
import os

path="./CDSdata"
fastas = glob.glob(os.path.join(path, 'GCF_000009045.1_ASM904v1_cds_from_genomic.fna'))
fastas


# In[75]:


path="./CDSdata"
fastas = glob.glob(os.path.join(path, 'GCF_000009045.1_ASM904v1_cds_from_genomic.fna'))
for fasta in fastas:
    print(fasta)


# In[1]:


import torch
map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
pretrain_path = "./Result/pretrain/protein_family.pt"
pt = torch.load(pretrain_path, map_location=map_location)


# In[ ]:




