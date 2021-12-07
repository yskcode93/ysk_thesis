import os
from dataclasses import dataclass
import psycopg2

from Bio import SeqIO
import pandas as pd
import re

# not safe
user = "admin"
password = "admin"
host = "postgres"
port = 5432
db = "admin"

# data class
@dataclass
class Species:
    id: int
    kingdom: str
    phylum: str
    _class: str
    order: str
    family: str
    genus: str
    species: str

    def __str__(self):
        return "species"

@dataclass
class Strain:
    id: int
    species_id: int
    name: str
    assembly_id: str

    def __str__(self):
        return "strain"

@dataclass
class Gene:
    id: int
    seq_aa: str
    seq_nc: str
    length_aa: int
    length_nc: int
    protein_id: str
    protein_type: str
    locus_tag: str
    _from: int
    to: int
    name: str
    strain_id: int

    def __str__(self):
        return "gene"

@dataclass
class Run_Strain:
    id: int
    run_id: int
    strain_id: int

    def __str__(self):
        return "run_strain"

@dataclass
class Gene_Gene:
    id: int
    run_id: int
    gene_1_id: int
    gene_2_id: int
    score: float
    length_ratio: float

    def __str__(self):
        return "gene_gene"

@dataclass
class Run:
    id: int
    name: str
    species_1_id: int
    species_2_id: int
    mode: str
    sensitivity: str
    complete_align: bool
    max_len_diff: float
    inflation: float
    single_linkage: bool
    max_gene_per_sp: int

    def __str__(self):
        return "run"

def get_connection(usr, pswd, host, port, db):
    return psycopg2.connect('postgresql://{}:{}@{}:{}/{}'.format(usr, pswd, host, port, db))

def quote_reserved(sql):
    reserved = ["order", "from", "to", "mode"]
    for res in reserved:
        sql = sql.replace("{}".format(res), "\"{}\"".format(res))
    return sql

def insert(obj):
    table = str(obj)
    keys = []
    values = []
    formats = []
    for k, v in vars(obj).items():
        if k=="id": continue
        if k.startswith("_"): k = k.replace("_", "")
        keys.append(k)
        values.append(v)
        formats.append("%s")

    sql = 'INSERT INTO {} ({}) VALUES ({}) RETURNING id;'.format(table, ", ".join(keys), ", ".join(formats))
    sql = quote_reserved(sql)

    with get_connection(user, password, host, port, db) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(values))

            (id,) = cur.fetchone()
            
        conn.commit()
        
    obj.id = id
    return obj

def fetchall(sql, params=None):
    with get_connection(user, password, host, port, db) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return rows

def execute(sql, params=None):
    with get_connection(user, password, host, port, db) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)

def get_gene(protein_id, strain_id):
    sql = 'SELECT id, length_aa, seq_aa FROM gene WHERE protein_id=%s AND strain_id=%s'

    with get_connection(user, password, host, port, db) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (protein_id, strain_id))
            (id, length_aa, seq_aa) = cur.fetchone()
            
    return id, length_aa, seq_aa

def insert_genes(fna, faa, strain):
    with open(fna, "r") as f:
        ncs = list(SeqIO.parse(f, "fasta"))

    with open(faa, "r") as f:
        aas = list(SeqIO.parse(f, "fasta"))

    for (aa, nc) in zip(aas, ncs):
        seq_aa = str(aa.seq)
        seq_nc = str(nc.seq)
        length_aa = len(seq_aa)
        length_nc = len(seq_nc)
        
        #name
        m = re.search(r"gene=[A-Za-z]+", aa.description)
        name = m.group().split("=")[1] if m else None
        # protein_id
        m = re.search(r"protein_id=[A-Z0-9_.]+", aa.description)
        protein_id = m.group().split("=")[1] if m else None
        # protein_type
        m = re.search(r"protein=[\S\s]+?]", aa.description)
        m = m.group().replace("[", "").replace("]", "")
        protein_type = m.split("=")[1] if m else None
        # locus tag
        m = re.search(r"locus_tag=[A-Z0-9_]+", aa.description)
        locus_tag = m.group().split("=")[1] if m else None
        # from, to
        m = re.search(r"location=[\S\s]+?]", aa.description)
        loc = m.group().replace("]", "").replace(">", "").replace("<", "").split("=")[1]
        m = re.search(r"\d+\.\.\d+", loc)
        start, end = m.group().split("..") if m else ("0", "0")
        start = int(start)
        end = int(end)
        (_from, to) = (end, start) if "complement" in loc else (start, end)
                
        gene = Gene(id=-1, seq_aa=seq_aa, seq_nc=seq_nc, length_aa=length_aa, length_nc=length_nc,\
                protein_id=protein_id, protein_type=protein_type, locus_tag=locus_tag, _from=int(_from), to=int(to),\
                name=name, strain_id=strain.id)
        
        gene = insert(gene)

def insert_orthologous_relations(species_1, species_2, strain_1, strain_2, fna_1, faa_1, fna_2, faa_2, run, orthogroup,\
    column_1, column_2):
    # insert species if there's no records
    sql = 'SELECT id FROM species WHERE species=%s'
    params = (species_1.species,)
    res = fetchall(sql, params)
    if len(res) > 0:
        species_1.id = res[0]
    else:
        species_1 = insert(species_1)

    params = (species_2.species,)
    res = fetchall(sql, params)
    if len(res) > 0:
        species_2.id = res[0]
    else:
        species_2 = insert(species_2)
    
    # insert strain if there's no records
    sql = 'SELECT id FROM strain WHERE assembly_id=%s'
    params = (strain_1.assembly_id,)
    res = fetchall(sql, params)
    if len(res) > 0:
        strain_1.id = res[0]
    else:
        strain_1.species_id = species_1.id
        strain_1 = insert(strain_1)
        # insert genes
        insert_genes(fna_1, faa_1, strain_1)
    
    params = (strain_2.assembly_id,)
    res = fetchall(sql, params)
    if len(res) > 0:
        strain_2.id = res[0]
    else:
        strain_2.species_id = species_2.id
        strain_2 = insert(strain_2)
        # insert genes
        insert_genes(fna_2, faa_2, strain_2)

    # insert run if there's no record
    sql = 'SELECT id FROM run WHERE id=%s'
    params = (run.id,)
    res = fetchall(sql, params)
    if len(res) > 0:
        run.id = res[0]
    else:
        run.species_1_id = species_1.id
        run.species_2_id = species_2.id
        run = insert(run)
        # insert run_strain
        run_strain_1 = Run_Strain(id=-1, run_id=run.id, strain_id=strain_1.id)
        run_strain_1 = insert(run_strain_1)
        run_strain_2 = Run_Strain(id=-1, run_id=run.id, strain_id=strain_2.id)
        run_strain_2 = insert(run_strain_2)
        # insert orthologous relation
        df = pd.read_csv(orthogroup, delimiter="\t")
        for src, tgt in df.loc[:, [column_1, column_2]].values:
            for s in src.split(","):
                for t in tgt.split(","):
                    pro_s = s.split("_prot_")[1]
                    pro_t = t.split("_prot_")[1]
                    m = re.search(r"[A-Z][A-Z]_\d+.\d", pro_s)
                    pro_id_s = m.group() if m else None
                    m = re.search(r"[A-Z][A-Z]_\d+.\d", pro_t)
                    pro_id_t = m.group() if m else None
                    
                    if pro_id_s and pro_id_t:
                        id_s, len_s, _ = get_gene(pro_id_s, strain_1.id)
                        id_t, len_t, _ = get_gene(pro_id_t, strain_2.id)
                    
                        gg = Gene_Gene(id=-1, run_id=run.id, gene_1_id=id_s, gene_2_id=id_t, score=0.0, length_ratio=len_t/len_s)
                        gg = insert(gg)