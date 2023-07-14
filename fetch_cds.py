import gzip
import pandas as pd
import os
import subprocess
from urllib.parse import urlparse, urlunparse

df = pd.read_csv("bacteria_complete_rep_ref_sellect.csv")
for url in df['RefSeq FTP']:
    o = urlparse(url)
    
    # path_fna = o.path + "/" + os.path.basename(url) + '_cds_from_genomic.fna.gz'
    # url_fna = urlunparse(o._replace(scheme='https',path=path_fna))
    path_faa = o.path + "/" + os.path.basename(url) + '_translated_cds.faa.gz'
    url_faa = urlunparse(o._replace(scheme='https',path=path_faa))

    # if os.path.exists(os.path.basename(os.path.basename(url) + '_cds_from_genomic.fna.gz')):
    #     continue
    if os.path.exists(os.path.basename(os.path.basename(url) + '_translated_cds.faa.gz')):
        continue

    print("downloading", url_faa)
    #command = f'curl -fsSL -O {url_faa}'
    command = f'curl -fsSL -O {url_fna} -O {url_faa}'
    command = command.split(' ')
    subprocess.run(command)
# !mv *.gz ../FULL/
#!gunzip -f *.gz
