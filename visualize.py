import matplotlib.pyplot as plt
import numpy as np

import re
from tqdm import tqdm

from .analysis import translation_map, alignment, identity, nc_similarity, amino_to_codon, LEVEL_AMINO

METRIC_IDENTITY_AA = 0
METRIC_IDENTITY_NC = 1
METRIC_COSINE = 2
METRIC_SCORE_AA = 3
METRIC_SCORE_NC = 4
METRIC_NAMES = [
    "Identity(aa)",
    "Identity(nc)",
    "Cosine similarity",
    "Alignment score(aa)",
    "Alignment score(nc)"
]
def scatter_plot(src, tgt, gen, out="", metric=METRIC_IDENTITY_AA, weight_path="./mlm_with_triplet.pt", lowest=0.0):
    if metric == METRIC_IDENTITY_AA:
        sts = []
        sgs = []
        gts = []
        for s, t, g in tqdm(zip(src, tgt, gen)):
            s, t, g = str(s.seq), str(t.seq), str(g.seq)
            sts.append(identity(alignment(s, t, LEVEL_AMINO)))
            sgs.append(identity(alignment(s, g, LEVEL_AMINO)))
            gts.append(identity(alignment(g, t, LEVEL_AMINO)))

        sts = np.array(sts)
        sgs = np.array(sgs)
        gts = np.array(gts)
    elif metric == METRIC_IDENTITY_NC: # based on reading frame
        sts = []
        sgs = []
        gts = []
        for s, t, g in tqdm(zip(src, tgt, gen)):
            s, t, g = str(s.seq), str(t.seq), str(g.seq)
            s_aa = "".join(map(lambda x: translation_map[x] if x in translation_map else "", re.split('(...)',s)[1:-1:2]))
            t_aa = "".join(map(lambda x: translation_map[x] if x in translation_map else "", re.split('(...)',t)[1:-1:2]))
            g_aa = "".join(map(lambda x: translation_map[x] if x in translation_map else "", re.split('(...)',g)[1:-1:2]))

            sts.append(nc_similarity(*amino_to_codon(s, t, alignment(s_aa, t_aa, LEVEL_AMINO))))
            sgs.append(nc_similarity(*amino_to_codon(s, g, alignment(s_aa, g_aa, LEVEL_AMINO))))
            gts.append(nc_similarity(*amino_to_codon(g, t, alignment(g_aa, t_aa, LEVEL_AMINO))))

        sts = np.array(sts)
        sgs = np.array(sgs)
        gts = np.array(gts)
    """
    elif metric == METRIC_COSINE:
        # load pretrain model
        model = gMLP(
            66,
            512,
            1024,
            700,
            32,
            nn.Tanh()
        )
        checkpoint = torch.load(weight_path)["model"]
        model.load_state_dict(checkpoint)
        model.cuda()

        # convert sequences to tensor
        codons = itertools.product("ATGC", repeat=3)
        codons = ["".join(c) for c in codons]
        vocab = ["<pad>"] + codons + ["<mask>"]

        tensor_s = []
        tensor_t = []
        tensor_g = []

        for s, t, g in zip(src, tgt, gen):
            s = re.split('(...)',str(s.seq))[1:-1:2]
            t = re.split('(...)',str(t.seq))[1:-1:2]
            g = re.split('(...)',str(g.seq))[1:-1:2]
            s = list(map(lambda x: vocab.index(x), s))
            t = list(map(lambda x: vocab.index(x), t))
            g = list(map(lambda x: vocab.index(x), g))
    
            s = torch.tensor(s)
            t = torch.tensor(t)
            g = torch.tensor(g)
            tensor_s.append(s)
            tensor_t.append(t)
            tensor_g.append(g)
        
        src = pad_sequence(tensor_s, batch_first=True)
        tgt = pad_sequence(tensor_t, batch_first=True)
        gen = pad_sequence(tensor_g, batch_first=True)

        if src.size(1) < 700:
            zeros = torch.zeros(src.size(0), 700-src.size(1)).long()
            src = torch.cat((src, zeros), dim=1)
        if tgt.size(1) < 700:
            zeros = torch.zeros(tgt.size(0), 700-tgt.size(1)).long()
            tgt = torch.cat((tgt, zeros), dim=1)
        if gen.size(1) < 700:
            zeros = torch.zeros(gen.size(0), 700-gen.size(1)).long()
            gen = torch.cat((gen, zeros), dim=1)

        sts = cosine_similarity(model.cuda(), src.cuda(), tgt.cuda()).cpu().detach().numpy()
        sgs = cosine_similarity(model.cuda(), src.cuda(), gen.cuda()).cpu().detach().numpy()
        gts = cosine_similarity(model.cuda(), gen.cuda(), tgt.cuda()).cpu().detach().numpy()

    elif metric == METRIC_SCORE_AA:
        pass
    elif metric == METRIC_SCORE_NC:
        pass
    """
        
    fig = plt.figure(figsize=(16,5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.plot(np.arange(lowest, 1.01, 0.01), np.arange(lowest, 1.01, 0.01), linestyle="dashed", color="gray")
    ax1.scatter(sts, gts, alpha=0.3)
    ax1.set_xlim(lowest, 1.0)
    ax1.set_ylim(lowest, 1.0)
    ax1.set_xlabel("{} between source and target".format(METRIC_NAMES[metric]))
    ax1.set_ylabel("{} between converted and target".format(METRIC_NAMES[metric]))
    ax1.text(0.95, 0.05, "{:.3f}({:.3f}) +/- {:.3f}({:.3f})".format(gts.mean(), sts.mean(), gts.std(), sts.std()),
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes,
        color='black', fontsize=12)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(np.arange(lowest, 1.01, 0.01), np.arange(lowest, 1.01, 0.01), linestyle="dashed", color="gray")
    ax2.scatter(sgs, gts, alpha=0.3)
    ax2.set_xlim(lowest, 1.0)
    ax2.set_ylim(lowest, 1.0)
    ax2.set_xlabel("{} between source and converted".format(METRIC_NAMES[metric]))
    ax2.set_ylabel("{} between converted and target".format(METRIC_NAMES[metric]))

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(np.arange(lowest, 1.01, 0.01), np.arange(lowest, 1.01, 0.01), linestyle="dashed", color="gray")
    ax3.scatter(sgs, sts, alpha=0.3)
    ax3.set_xlim(lowest, 1.0)
    ax3.set_ylim(lowest, 1.0)
    ax3.set_xlabel("{} between source and converted".format(METRIC_NAMES[metric]))
    ax3.set_ylabel("{} between source and target".format(METRIC_NAMES[metric]))
    
    plt.title("{:.3f} +/- {:.3f}".format(gts.mean(), gts.std()))

    plt.savefig(out)
    plt.clf()
    plt.close()

    return sts, gts, sgs