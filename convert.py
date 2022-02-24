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
from .file import GeneBatchLoader, sanitize, vocab, write_fna_faa
# fetch all
from .database import fetchall

# train test split
from sklearn.model_selection import KFold

# base package
import re
import time
import numpy as np
from tqdm import tqdm
from Bio import SeqIO

# inference mode
MODE_GREEDY = 0
MODE_BEAM = 1

def cross_validate(gpu, world_size, ConverterClass, config, weight_path, n_epochs, batch_size, lr, warmup, use_apex, finetune,\
        strain_ids, run_ids, direction, mode,\
        pretrain, PretrainClass, config_pretrain, pretrain_path,\
        log_interval=1, save_interval=1, output_dir="./Result", random_state=0\
    ):
    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)
    # 
    sql = """
        SELECT g1.seq_nc, g2.seq_nc, g1.strain_id, g2.strain_id, g1.id, g2.id FROM gene_gene as gg
        INNER JOIN gene AS g1 ON gg.gene_1_id=g1.id
        INNER JOIN gene AS g2 ON gg.gene_2_id=g2.id
        WHERE gg.run_id IN ({}) AND g1.length_nc <= 2100 AND g2.length_nc <= 2100 AND gg.length_ratio BETWEEN 0.97 AND 1.03;
    """.format(",".join(["%s"]*len(run_ids)))

    X, Y, X_sp, Y_sp, X_id, Y_id = [], [], [], [], [], []

    for x, y, x_sp, y_sp, x_id, y_id in fetchall(sql, run_ids):
        x, y = sanitize(x), sanitize(y)
        x, y = re.split('(...)',x)[1:-1:2], re.split('(...)',y)[1:-1:2]
        x, y = list(map(lambda x: vocab.index(x), x)), list(map(lambda x: vocab.index(x), y))
        if pretrain and "n_species" in config_pretrain:
            x, y = [len(vocab)] + x, [len(vocab)] + y
        x, y = torch.tensor(x), torch.tensor(y)
        x_sp, y_sp = 0, 1#strain_ids.index(x_sp), strain_ids.index(y_sp)

        if direction==0:
            X += [x, y]
            Y += [y, x]
            X_sp += [x_sp, y_sp]
            Y_sp += [y_sp, x_sp]
            X_id += [x_id, y_id]
            Y_id += [y_id, x_id]
        elif direction==1:
            X.append(x)
            Y.append(y)
            X_sp.append(x_sp)
            Y_sp.append(y_sp)
            X_id.append(x_id)
            Y_id.append(y_id)
        elif direction==2:
            X.append(y)
            Y.append(x)
            X_sp.append(y_sp)
            Y_sp.append(x_sp)
            X_id.append(y_id)
            Y_id.append(x_id)

    X, Y = pad_sequence(X, batch_first=True), pad_sequence(Y, batch_first=True)
    X_sp, Y_sp = torch.tensor(X_sp).unsqueeze(1), torch.tensor(Y_sp).unsqueeze(1)
    X_id, Y_id = np.array(X_id), np.array(Y_id)

    max_length = 701 if pretrain and "n_species" in config_pretrain else 700

    if X.size(1) < max_length:
        X = torch.cat((X, torch.zeros(X.size(0), max_length-X.size(1)).long()), dim=1)

    if Y.size(1) < max_length:
        Y = torch.cat((Y, torch.zeros(Y.size(0), max_length-Y.size(1)).long()), dim=1)

    # pretrain model
    if pretrain:
        model_pretrain = PretrainClass(**config_pretrain).to(gpu)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        pt = torch.load(pretrain_path, map_location=map_location)
        model_pretrain.load_state_dict(pt["model"])
        model_pretrain.eval()

    # cross validation
    kf = KFold(random_state=random_state, shuffle=True)
    Y_id_set = np.unique(Y_id)
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(Y_id_set)):
        Y_id_set_test = Y_id_set[test_idx]
        test_idx = np.isin(Y_id, Y_id_set_test)
        train_idx = ~ test_idx

        # write only source and target
        write_fna_faa(X[test_idx, 1:] if pretrain and "n_species" in config_pretrain else X[test_idx], "{}/finetune/src_{}".format(output_dir, fold_idx+1), "SRC")
        write_fna_faa(Y[test_idx, 1:] if pretrain and "n_species" in config_pretrain else Y[test_idx], "{}/finetune/tgt_{}".format(output_dir, fold_idx+1), "TGT")

        dataset = dat.TensorDataset(X[train_idx], Y[train_idx], Y_sp[train_idx])

        # load pretrained converter
        model = ConverterClass(**config).to(gpu)
        if finetune:
            dist.barrier()
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            pt = torch.load(weight_path, map_location=map_location)
            model.load_state_dict(pt["model"])
            model.freeze()

        optimizer = opt.AdamW(model.parameters())
        criterion = nn.CrossEntropyLoss()

        if use_apex:
            model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

        model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu], find_unused_parameters=True)
        # loader
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank
        )
        loader = dat.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            sampler=sampler
        )

        # scheduler
        scheduler = lrs.OneCycleLR(
            optimizer,
            lr,
            epochs = n_epochs,
            steps_per_epoch = len(loader),
            pct_start=warmup,
            anneal_strategy='linear'
        )

        for i in range(1, n_epochs+1):
            start = time.time()

            sampler.set_epoch(i)

            avg_loss = 0

            for x, y, y_sp in loader:
                optimizer.zero_grad()
                x, y, y_sp = x.cuda(non_blocking=True), y.cuda(non_blocking=True), y_sp.cuda(non_blocking=True)

                if pretrain:
                    with torch.no_grad():
                        x_e = model_pretrain.get_output(x)

                    out = model(x_e, x[:,1:], y_sp, y[:,1:]) if pretrain and "n_species" in config_pretrain else model(x_e, x, y_sp, y)
                else:
                    out = model(x, x, y_sp, y)

                loss = criterion(out.permute(0,2,1), y[:, 1:] if pretrain and "n_species" in config_pretrain else y)
                if use_apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # prevent gradient explosion
                utils.clip_grad_norm_(model.parameters(), 1)

                avg_loss += loss.item()

                optimizer.step()
                scheduler.step()

            avg_loss /= len(loader)

            if rank == 0 and i%log_interval==0:
                with open("{}/finetune.log".format(output_dir), "a") as f:
                    f.write("FOLD {} EPOCH({:0=3}%) LOSS: {:.4f} lr={:.4f} x 1e-4 ELAPSED TIME: {:.4f}s\n".format(fold_idx+1, int(i*100/n_epochs), avg_loss,\
                        scheduler.get_last_lr()[0]*1e4, time.time()-start))

            if rank == 0 and (i==0 or i%save_interval==0):
                checkpoint = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'amp': amp.state_dict(),
                    'scheduler': scheduler.state_dict()
                } if use_apex else {
                    'model': model.module.state_dict(),
                    'scheduler': scheduler.state_dict()
                }

                torch.save(checkpoint, "{}/weight/checkpoint_{}_{}.pt".format(output_dir, fold_idx+1, i))

def test_from_fasta(ConverterClass, config, strain_ids, sp, n_epochs, n_folds, mode, device,\
    pretrain, PretrainClass, config_pretrain, pretrain_path,\
    output_dir="./Result", batch_size=100):
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
        pt = torch.load("{}/weight/checkpoint_{}.pt".format(output_dir, n_epochs), map_location=lambda storage, loc: storage.cuda(device))
        model = ConverterClass(**config).to(device)
        model.load_state_dict(pt["model"])
        model.eval()
        del pt

        # pretrain class
        if pretrain:
            pt = torch.load(pretrain_path, map_location=lambda storage, loc: storage.cuda(device))
            model_pretrain = PretrainClass(**config_pretrain).to(device)
            model_pretrain.load_state_dict(pt["model"])
            model_pretrain.eval()
            del pt

        with torch.no_grad():
            if pretrain:
                cls = torch.tensor([[len(vocab)]]*X.size(0)).to(device)
                X_e = model_pretrain.get_output(torch.cat((cls, X), dim=1)) if "n_species" in config_pretrain else model_pretrain.get_output(X)
            else:
                X_e = X

            if mode == MODE_GREEDY:
                out = model.infer_greedily(X_e, X, Y_sp, X.size(1), [17, 19, 25])
            elif mode == MODE_BEAM:
                outs = []
                for x_e, x, y_sp in zip(torch.split(X_e, batch_size), torch.split(X, batch_size), torch.split(Y_sp, batch_size)):
                    out = model.beam_search(x_e, x, y_sp, x.size(1), 5, [17, 19, 25])
                    if out.size(1) < 700:
                        out = torch.cat((out.cpu(), torch.zeros(out.size(0), 700-out.size(1)).long()), dim=1)
                    outs.append(out.cpu())

                out = torch.cat(outs)

        write_fna_faa(out.cpu(), "{}/finetune/gen_{}".format(output_dir, i), "GEN")        

def test(ConverterClass, config, strain_ids, run_ids, direction, n_epochs, n_samples, mode, device, output_dir="./Result"):
    pt = torch.load("{}/weight/checkpoint_{}.pt".format(output_dir, n_epochs), map_location=lambda storage, loc: storage.cuda(device))
    model = ConverterClass(**config).to(device)
    model.load_state_dict(pt["model"])
    model.eval()
    del pt

    # 
    sql = """
        SELECT g1.seq_nc, g2.seq_nc, g1.strain_id, g2.strain_id FROM gene_gene as gg
        INNER JOIN gene AS g1 ON gg.gene_1_id=g1.id
        INNER JOIN gene AS g2 ON gg.gene_2_id=g2.id
        WHERE gg.run_id IN ({}) AND g1.length_nc <= 2100 AND g2.length_nc <= 2100 AND gg.length_ratio BETWEEN 0.97 AND 1.03;
    """.format(",".join(["%s"]*len(run_ids)))

    X, Y, X_sp, Y_sp = [], [], [], []

    for x, y, x_sp, y_sp in fetchall(sql, run_ids):
        x, y = sanitize(x), sanitize(y)
        x, y = re.split('(...)',x)[1:-1:2], re.split('(...)',y)[1:-1:2]
        x, y = list(map(lambda x: vocab.index(x), x)), list(map(lambda x: vocab.index(x), y))
        x, y = torch.tensor(x), torch.tensor(y)
        x_sp, y_sp = 0,1#strain_ids.index(x_sp), strain_ids.index(y_sp)

        if direction==0:
            X += [x, y]
            Y += [y, x]
            X_sp += [x_sp, y_sp]
            Y_sp += [y_sp, x_sp]
        elif direction==1:
            X.append(x)
            Y.append(y)
            X_sp.append(x_sp)
            Y_sp.append(y_sp)
        elif direction==2:
            X.append(y)
            Y.append(x)
            X_sp.append(y_sp)
            Y_sp.append(x_sp)

    X, Y = pad_sequence(X, batch_first=True), pad_sequence(Y, batch_first=True)
    X_sp, Y_sp = torch.tensor(X_sp).unsqueeze(1), torch.tensor(Y_sp).unsqueeze(1)

    if X.size(1) < 700:
        X = torch.cat((X, torch.zeros(X.size(0), 700-X.size(1)).long()), dim=1)

    if Y.size(1) < 700:
        Y = torch.cat((Y, torch.zeros(Y.size(0), 700-Y.size(1)).long()), dim=1)

    # sample random sequences
    idx = torch.randint(X.size(0), size=(n_samples,))
    X, Y, X_sp, Y_sp = X[idx], Y[idx], X_sp[idx], Y_sp[idx]

    write_fna_faa(X, "{}/test/beam_search/src_{}".format(output_dir, n_epochs), "SRC")
    write_fna_faa(Y, "{}/test/beam_search/tgt_{}".format(output_dir, n_epochs), "TGT")

    with torch.no_grad():
        X, Y_sp = X.to(device), Y_sp.to(device)
        #model(X, X, Y_sp)
        if mode == MODE_GREEDY:
            out = model.infer_greedily(X, X, Y_sp, X.size(1), [17, 19, 25])
        elif mode == MODE_BEAM:
            out = model.beam_search(X, X, Y_sp, X.size(1), 5, [17, 19, 25])

    write_fna_faa(out.cpu(), "{}/test/beam_search/gen_{}".format(output_dir, n_epochs), "GEN")

# This function is only for training
# direction = {0: both, 1: x -> y, 2: y-> x}
def train(gpu, world_size, ConverterClass, config, n_epochs, batch_size, lr, warmup, use_apex,\
        strain_ids, gene_batch_size, total_count, direction,\
        pretrain, PretrainClass, config_pretrain, pretrain_path,\
        checkpoint,\
        log_interval=10, save_interval=100, output_dir="./Result"\
    ):
    # rank equals gpu when using one node only
    rank = gpu
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    torch.manual_seed(0)
    torch.cuda.set_device(gpu)

    # define model
    model = ConverterClass(**config).to(gpu)
    optimizer = opt.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    if use_apex:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')

    # scheduler
    scheduler = lrs.OneCycleLR(
        optimizer,
        lr,
        total_steps=int(float(total_count)*n_epochs/(batch_size*world_size)) if direction>0 else int(float(total_count)*n_epochs*2/(batch_size*world_size)),
        pct_start=warmup,
        anneal_strategy='linear'
    )

    # resume training
    if checkpoint > 0:
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        chkpt = torch.load("{}/checkpoint_{}.pt".format(output_dir, checkpoint), map_location=map_location)
        model.load_state_dict(chkpt["model"])

        if use_apex:
            amp.load_state_dict(chkpt["amp"])
            optimizer.load_state_dict(chkpt["optimizer"])
            scheduler.load_state_dict(chkpt["scheduler"])
        else:
            scheduler.load_state_dict(chkpt["scheduler"])

    if pretrain:
        pretrained_model = PretrainClass(**config_pretrain).to(gpu)
        dist.barrier()
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        chkpt = torch.load(pretrain_path, map_location=map_location)
        pretrained_model.load_state_dict(chkpt["model"])
        pretrained_model.eval()

    model = DDP_APEX(model) if use_apex else DDP(model, device_ids=[gpu], find_unused_parameters=True)

    # data loader
    sql = """
        SELECT gg.id, g1.seq_nc, g2.seq_nc, g1.strain_id, g2.strain_id FROM gene_gene as gg
        INNER JOIN gene AS g1 ON gg.gene_1_id=g1.id
        INNER JOIN gene AS g2 ON gg.gene_2_id=g2.id
        WHERE gg.run_id IN (549, 548) AND g1.length_nc <= 2100 AND g2.length_nc <= 2100 AND gg.length_ratio BETWEEN 0.97 AND 1.03 AND gg.id > %s
        ORDER BY gg.id FETCH FIRST %s ROWS ONLY;
    """

    X_, Y_, X_sp_, Y_sp_ = [], [], [], []
    for t in tqdm(range(2, 6)):
        # source sequences
        with open("{}/finetune/src_{}.fna".format(output_dir, t), "r") as f:
            x = list(SeqIO.parse(f, "fasta"))

        x = [sanitize(str(seq.seq)) for seq in x]
        x = [re.split('(...)',seq)[1:-1:2] for seq in x]
        x = [torch.tensor(list(map(lambda x: vocab.index(x), seq))) for seq in x]
        X_ += x

        # target sequences
        with open("{}/finetune/tgt_{}.fna".format(output_dir, t), "r") as f:
            y = list(SeqIO.parse(f, "fasta"))

        y = [sanitize(str(seq.seq)) for seq in y]
        y = [re.split('(...)',seq)[1:-1:2] for seq in y]
        y = [torch.tensor(list(map(lambda x: vocab.index(x), seq))) for seq in y]
        Y_ += y

        X_sp_ += [1]*len(x)
        Y_sp_ += [0]*len(y)

    count = len(X_)
    
    for i in range(1, n_epochs+1):
        # start time
        start = time.time()
        gbl = GeneBatchLoader(sql, gene_batch_size, total_count)

        # average loss
        avg_loss = 0

        for j, rows in enumerate(gbl):
            X, Y, X_sp, Y_sp = X_, Y_, X_sp_, Y_sp_
            del X[count:], Y[count:], X_sp[count:], Y_sp[count:]
            #X, Y, X_sp, Y_sp = [], [], [], []
            for last_id, x, y, x_sp, y_sp in rows:
                x, y = sanitize(x), sanitize(y)
                x, y = re.split('(...)',x)[1:-1:2], re.split('(...)',y)[1:-1:2]
                x, y = list(map(lambda x: vocab.index(x), x)), list(map(lambda x: vocab.index(x), y))
                if pretrain and "n_species" in config_pretrain:
                    x, y = [len(vocab)] + x, [len(vocab)] + y
                x, y = torch.tensor(x), torch.tensor(y)
                x_sp, y_sp = 0, 1#strain_ids.index(x_sp), strain_ids.index(y_sp)
                if direction==0:
                    X += [x, y]
                    Y += [y, x]
                    X_sp += [x_sp, y_sp]
                    Y_sp += [y_sp, x_sp]
                elif direction==1:
                    X.append(x)
                    Y.append(y)
                    X_sp.append(x_sp)
                    Y_sp.append(y_sp)
                elif direction==2:
                    X.append(y)
                    Y.append(x)
                    X_sp.append(y_sp)
                    Y_sp.append(x_sp)
            
            gbl.last_id = last_id

            X, Y = pad_sequence(X, batch_first=True), pad_sequence(Y, batch_first=True)
            X_sp, Y_sp = torch.tensor(X_sp).unsqueeze(1), torch.tensor(Y_sp).unsqueeze(1)

            max_length = 701 if pretrain and "n_species" in config_pretrain else 700

            if X.size(1) < max_length:
                X = torch.cat((X, torch.zeros(X.size(0), max_length-X.size(1)).long()), dim=1)

            if Y.size(1) < max_length:
                Y = torch.cat((Y, torch.zeros(Y.size(0), max_length-Y.size(1)).long()), dim=1)

            dataset = dat.TensorDataset(X, Y, Y_sp)
            
            # loader
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank
            )
            loader = dat.DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                sampler=sampler,
                drop_last = True
            )

            # different ordering for each epoch
            sampler.set_epoch(i)

            for x, y, y_sp in loader:
                optimizer.zero_grad()
                x, y, y_sp = x.cuda(non_blocking=True), y.cuda(non_blocking=True), y_sp.cuda(non_blocking=True)
                
                if pretrain:
                    with torch.no_grad():
                        x_e = pretrained_model.get_output(x)
                else:
                    x_e = x

                out = model(x_e, x[:,1:], y_sp, y[:,1:]) if pretrain and "n_species" in config_pretrain else model(x_e, x, y_sp, y)
                loss = criterion(out.permute(0,2,1), y[:,1:] if pretrain and "n_species" in config_pretrain else y)

                if use_apex:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # prevent gradient explosion
                utils.clip_grad_norm_(model.parameters(), 1)

                avg_loss += loss.item()

                optimizer.step()
                scheduler.step()

        avg_loss /= len(gbl) * len(loader)

        if rank == 0 and i%log_interval==0:
            with open("{}/train.log".format(output_dir), "a") as f:
                f.write("EPOCH({:0=3}%) LOSS: {:.4f} lr={:.4f} x 1e-4 ELAPSED TIME: {:.4f}s\n".format(int(i*100/n_epochs), avg_loss,\
                    scheduler.get_last_lr()[0]*1e4, time.time()-start))

        if rank == 0 and (i==1 or i%save_interval==0):
            write_fna_faa(x[:,1:].cpu() if pretrain and "n_species" in config_pretrain else x.cpu(), "{}/train/src_{}".format(output_dir, i), "SRC")
            write_fna_faa(y[:,1:].cpu() if pretrain and "n_species" in config_pretrain else y.cpu(), "{}/train/tgt_{}".format(output_dir, i), "TGT")
            write_fna_faa(torch.argmax(out, dim=2).cpu(), "{}/train/gen_{}".format(output_dir, i), "GEN")

            checkpoint = {
                'model': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'amp': amp.state_dict(),
                'scheduler': scheduler.state_dict()
            } if use_apex else {
                'model': model.module.state_dict(),
                'scheduler': scheduler.state_dict()
            }

            torch.save(checkpoint, "{}/weight/checkpoint_{}.pt".format(output_dir, i))

    dist.destroy_process_group()