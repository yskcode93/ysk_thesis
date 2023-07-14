import math
from functools import reduce
import torch
import torch.nn as nn
from torch import Tensor  # for type hint
import torch.nn.functional as F
import re
import pathlib
from file import PAD_RESTYPES_GAP,to_PAD_BASE_GAP_INDEX
from analysis import translation_map,start_map

#Path
CWD = pathlib.Path()
CLIENT_PATH = CWD / 'results' / 'mlruns.db'
MODEL_DIR = CWD / 'models'
DATA_DIR = CWD / 'data' / 'processed'

class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer Encoder.
    
    """
    def __init__(self, d_model: int, dropout: float = 0.10, max_len: int = 5000): # dropout: float = 0.1
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1,max_len, d_model)
        pe[0,:, 0::2] = torch.sin(position * div_term)
        pe[0,:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0) peの最初に0入れないならこれ
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)
        # AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE によるとここはdropoutするべき。
        # https://deepsquare.jp/2020/10/vision-transformer/

class Transformer(nn.Module):
    """Transformer Encoder model.
    
    """

    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        dropout_p,
        dropout_e,
        max_len,
        dff,
        pretrain:bool = False,
    ):
    #     dim_encoder:int = 128,
    #     dim_decoder:int = 128,
    # ):
        super().__init__()

        # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model
        self.pretrain = pretrain

        # LAYERS
        # if pretrain:
        #    self.change_dim = nn.Linear(dim_encoder,dim_model)
        # else:
        if not pretrain:
            self.embedding = nn.Embedding(num_tokens, dim_model) # https://gotutiyan.hatenablog.com/entry/2020/09/02/200144
        
        self.positional_encoder = PositionalEncoding(
            d_model=dim_model,
            dropout=dropout_p,
            max_len=max_len
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = dim_model,
            dim_feedforward = dff,
            dropout=dropout_e,
            nhead = num_heads,
            batch_first=True,
        )
        # norm_first=True Pre-LNは学習は簡単だが、精度は低い......
        # dropoutはdefaultで0.1
        # encoder_norm = nn.LayerNorm(dim_model)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        ) # enabled_nested_tensor Trueにするとpadding多い時にいいらしい。https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
        # if pretrain:
        #     self.to_decoder = nn.Linear(dim_model,dim_decoder)
        self.out = nn.Linear(dim_model, num_tokens)
        
    def forward(self, src, src_key_padding_mask = None):
        if src_key_padding_mask is None:
            src_key_padding_mask = (src == 0) # PAD_IDX = 0
        
        # if src.dim() == 3:
        #     src = self.change_dim(src)
        
        if not self.pretrain: #src.dim() == 2
            # Src size must be (batch_size, src sequence length)
            # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
            src = self.embedding(src) * math.sqrt(self.dim_model) # [batch_size, src sequence length, dim_model]
        
        src = self.positional_encoder(src) # [batch_size, sequence length, dim_model]
        # If Src size is (batch_size, sequence length, dim_model), it is already embedded vector.
        # Transformer blocks - Out size = (batch_size, sequence length, num_tokens)
        transformer_out = self.transformer_encoder(src,src_key_padding_mask=src_key_padding_mask) # [batch_size, sequence length, dim_model]
        
        # if self.pretrain:
        #     transformer_out = self.to_decoder(transformer_out)
        
        out = self.out(transformer_out) # ここはtrainedではnn.Identityに変更
        
        return out

# helpers
def prob_mask_like(t, prob):
    """ calcurate possibility of masking for every single element.
    """
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    """ tokens(pad,cls,sep) is True and tokens which can be masked is False.
    
    """
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


# main class
class MLM(nn.Module):
    """
        refs:
        - https://github.com/lucidrains/mlm-pytorch
    """
    def __init__(
        self,
        pretrain_model,
        mask_prob = 0.15,
        replace_prob = 0.8,
        rand_high_token = None, # 
        random_token_prob = 0.1,
        mask_token_id = 65, # Noneにしてnum_okenと同じようにassertかけるべき。
        pad_token_id = 0,
        mask_ignore_token_ids = []):
        super().__init__()

        self.pretrain_model = pretrain_model

        # mlm related probabilities

        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.rand_high_token = rand_high_token # 65 (1~64 codon をランダムで追加する。)
        self.random_token_prob = random_token_prob

        # token ids

        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

    def forward(self, seq):

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random

        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids) # mask_ignore_token_idsの部分をTrue
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob) # 15%の予測部分をTrue

        # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)

        masked_seq = seq.clone().detach()

        # derive labels to predict

        labels = seq.masked_fill(~mask, self.pad_token_id) # 予測しない部分のトークンを0(pad)に変更

        # [MASK] input

        replace_prob = prob_mask_like(seq, self.replace_prob) # 80%の確率でTrue (maskでもTrueならば、[MASK]トークンになる。)
        masked_seq = masked_seq.masked_fill(mask * replace_prob, self.mask_token_id) # [batch size, seq length]
        mask = mask & ~replace_prob # 予測部位(mask)のうち[MASK]ではない部分をTrue

        # if random token probability > 0 for mlm

        if self.random_token_prob > 0:
            assert self.rand_high_token is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
            # 予測トークンの10%をランダムトークンに変更。<=> 予測部位のうち[MASK]トークンではない部分(この時点で0.20かかっているので)を50%の確率でランダムトークンに置き換え。
            random_token_prob = prob_mask_like(seq, (self.random_token_prob/(1 - self.replace_prob))) 

            # [pad],[mask]を除いた1~64までのランダムトークン列　[そのまま]もできれば除きたい。0.1%くらい変わってきてしまうので。
            random_tokens = torch.randint(1, self.rand_high_token, seq.shape, device=seq.device)
            # random_tokens = torch.randint(1, 65, seq.shape, device=seq.device)
            random_token_prob &= mask # 予測部位のうち、ランダムトークンに置き換える部位をTrue
            
            # 1~64までのコドンで置き換え。
            masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)

            # remove tokens that were substituted randomly from being [mask]ed later
            # mask = mask & ~random_token_prob # 予測部位 かつ ランダム置き換え無し -> [MASK] or そのまま

        # if self.random_token_prob > 0:
        #     assert self.num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
        #     random_token_prob = prob_mask_like(seq, self.random_token_prob) # 10%で置き換え部位
        #     random_tokens = torch.randint(1, self.num_tokens, seq.shape, device=seq.device) # random token列を生成 いつも同じになってたりしないか心配
        #     # [pad],[mask],[そのまま] を除いたidに変更
        #     random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids) # mask_ignore_token_idsの部分をTrue
        #     random_token_prob &= ~random_no_mask # (確率的に決定された)置き換え部位 かつ 置き換えできるidならTrue
        #     # 1~64までのコドンで置き換え。
        #     masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)

            # remove tokens that were substituted randomly from being [mask]ed later
            # mask = mask & ~random_token_prob # 予測部位 かつ ランダム置き換え無し -> [MASK] or そのまま
        # # [mask] input

        # replace_prob = prob_mask_like(seq, self.replace_prob) # 80%の確率でTrue (maskでもTrueならば、[MASK]トークンになる。)
        # masked_seq = masked_seq.masked_fill(mask * replace_prob, self.mask_token_id) # [batch size, seq length]
        
        # get generator output and get mlm loss
        
        logits = self.pretrain_model(masked_seq)
        logits = logits.permute(0,2,1) # ->[batch_size, classes=n_tokens, dim=seq_length]

        mlm_loss = F.cross_entropy(
            logits,
            labels,
            ignore_index = self.pad_token_id
        )
           
        _,pred_codon = torch.max(input=logits,dim=1) # cpuに移動
        
        mlm_num = torch.count_nonzero(labels)
        # print(f'mask positions : {torch.count_nonzero(labels)}') # lossを計算している入力トークンの数
        # print(f'[MASK] tokens : {torch.count_nonzero(masked_seq == 65)}') # [MASK]トークンの数
        # print(f'MASK rate : {torch.count_nonzero(masked_seq == 65)/torch.count_nonzero(labels)}') # [MASK]トークンの割合 nCk 0.8^k 0.2^n-k
        # print(f'[RND] rate : {torch.count_nonzero(random_token_prob)/torch.count_nonzero(labels)}')

        correct_num = torch.count_nonzero((torch.logical_and(pred_codon==labels, labels!=0))) # 最大でbatch_size*seq_length*0.15
        
        # pred_codon==labels & labels!=0　でも同じじゃない？
        
        return mlm_loss, correct_num, mlm_num

class trained_Transformer(nn.Module):
    def __init__(
        self,
        encoder_run_id,
        transformer_config,
        decoder_run_id):
        super().__init__()
        pretrained_encoder = fetch_model(encoder_run_id)
        pretrained_encoder.out = nn.Identity()
        self.pretrained_encoder = pretrained_encoder
        
        transformer = Transformer(**transformer_config)
        transformer.out = nn.Identity()
        self.transformer = transformer
        
        pretrained_decoder = fetch_model(decoder_run_id)
        self.pretrained_decoder = pretrained_decoder.out
        
    def forward(self,src):
        src_key_padding_mask = (src == 0) # PAD_IDX = 0
        # Src size must be (batch_size, src sequence length)
        src = self.pretrained_encoder(src)
        # Pretrained Encoder - Out size = (batch_size, sequence length, dim_model)
        src = self.transformer(src.detach(), src_key_padding_mask) # ここをoptimzerに渡すので学習部分はif文使って中に埋め込む。
        # Trainable Encoder - Out size = (batch_size, sequence length, dim_model)
        out = self.pretrained_decoder(src)
        # Pretrained Decoder - Out size = (batch_size, sequence length, num_tokens)
        return out

class Transformer_with_trained_encoder(nn.Module):
    def __init__(
        self,
        encoder_run_id,
        transformer_config,):
        super().__init__()

        pretrained_encoder = fetch_model(encoder_run_id)
        pretrained_encoder.out = nn.Identity()
        self.pretrained_encoder = pretrained_encoder
        self.transformer = Transformer(**transformer_config)
    def forward(self,src):
        src_key_padding_mask = (src == 0) # PAD_IDX = 0
        # Src size must be (batch_size, src sequence length)
        src = self.pretrained_encoder(src)
        # Pretrained Encoder - Out size = (batch_size, sequence length, dim_model)
        out = self.transformer(src.detach(), src_key_padding_mask) # ここをoptimzerに渡すので学習部分はif文使って中に埋め込む。
        # Transformer - Out size = (batch_size, sequence length, num_tokens)
        return out

class Transformer_with_trained_encoder_trained_decoder_layers(nn.Module):
    def __init__(
        self,
        encoder_run_id,
        transformer_config,
        decoder_run_id,
        decoder_layer_num:int):
        super().__init__()

        pretrained_encoder = fetch_model(encoder_run_id)
        pretrained_encoder.out = nn.Identity()
        self.pretrained_encoder = pretrained_encoder
        
        transformer = Transformer(**transformer_config)
        transformer.out = nn.Identity()
        self.transformer = transformer

        pretrained_decoder = fetch_model(decoder_run_id)
        self.decoder_layers = pretrained_decoder.transformer_encoder.layers[-decoder_layer_num:] # n層のtransformer
        self.out = pretrained_decoder.out
        
    def forward(self,src):
        src_key_padding_mask = (src == 0) # PAD_IDX = 0
        # Src size must be (batch_size, src sequence length)
        src = self.pretrained_encoder(src)
        # Pretrained Encoder - Out size = (batch_size, sequence length, dim_model)
        src = self.transformer(src.detach(), src_key_padding_mask) # ここをoptimzerに渡すので学習部分はif文使って中に埋め込む。
        # Trainable Encoder - Out size = (batch_size, sequence length, dim_model)
        
        for delayer in self.decoder_layers:
            src = delayer(src)
        out = self.out(src)

        return out

class Transformer_representative_input(nn.Module):
    def __init__(
        self,
        transformer_config,
        decoder_run_id):
        super().__init__()
        
        transformer = Transformer(**transformer_config) # self.pretrainが設定されていればこれで急に埋め込み表現入れられるはず。
        transformer.out = nn.Identity()
        self.transformer = transformer
        
        pretrained_decoder = fetch_model(decoder_run_id)
        self.pretrained_decoder = pretrained_decoder.out
        
    def forward(self,src,src_representative):
        src_key_padding_mask = (src == 0) # PAD_IDX = 0
        # Src size must be (batch_size, src sequence length)
        src = self.transformer(src_representative.detach(), src_key_padding_mask) # ここをoptimzerに渡すので学習部分はif文使って中に埋め込む。
        # Trainable Encoder - Out size = (batch_size, sequence length, dim_model)
        out = self.pretrained_decoder(src)
        # Pretrained Decoder - Out size = (batch_size, sequence length, num_tokens)
        return out
          

def get_key_from_value(d, val):
    keys = [k for k, v in d.items() if v == val]
    if keys:
        return keys
    return None

def codon_prob_matrics(num_tokens:int = 66,epsilon:float = 0.1,use_start_codon:bool = False):
    matrics = torch.zeros(num_tokens, num_tokens)
    for i in range(num_tokens):
        codon = PAD_RESTYPES_GAP[i] if i < len(PAD_RESTYPES_GAP) else ''
        aminoacid = translation_map.get(codon)
        codons = get_key_from_value(translation_map,aminoacid)
        sim_tokens = [PAD_RESTYPES_GAP.index(x) for x in codons] if codons is not None else [i]
        # start codonを反映する場合
        if use_start_codon:
            start_amino = start_map.get(codon)
            start_codons = get_key_from_value(start_map,start_amino)
            sim_tokens += [PAD_RESTYPES_GAP.index(x) for x in start_codons] if start_codons is not None else [i]
            sim_tokens = list(set(sim_tokens))
        # 1-e + e/n(正解になる可能性のあるラベル)
        possible_label_num = len(sim_tokens)
        #matrics[i][i] = 1-epsilon
        matrics[i][sim_tokens] += 1/possible_label_num
    return matrics

class cls_CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        device,
        epsilon:float = 0.0,
        num_classes:int = 66, #65 pad codon ,# 66 pad codon msk ,# 67 pad codon gap mask
        use_start_codon:bool = False):
        super().__init__()

        self.device = device
        self.matrics = codon_prob_matrics(num_tokens=num_classes,epsilon=epsilon,use_start_codon=use_start_codon)
    def forward(self, input, target):
        target = self.matrics[target] # matricsのサイズは出力クラス^2でなければならない
        target = target.permute(0,2,1)
        target = target.to(self.device)
        loss = F.cross_entropy(input, target)
        return loss

class ThreeBaseLoss(nn.Module):
    def __init__(
        self,
        device,
        fst):
        super().__init__()

        self.device = device
        self.fst = fst

    def forward(self, input, target):
        """
        Args:
            input: batch_size, classes(~66), seq_length -> input: batch_size, classes(~6), seq_length 
            target: batch_size, dim=seq_length
        Returns:
            loss: three residue loss
        """
        def to_base(seqs ,position:int):
            """seqs to base at the given position. e.g. ATG(33),1 -> A(1) pad,a,c,g,t
            Args:
                seqs: batch_size, classes, seq_length.
            """
            if position <= 3:
                base = torch.from_numpy(to_PAD_BASE_GAP_INDEX(position))[seqs]
            else: raise ValueError('position is less than 3.')
            
            return base
        
        def to_base_prob(codon_prob, position:int,device):
            """summation of the same base probabilities in codon classification.
            Args: 
                codon_prob: batch_size, 65(w/o p) or 66(w/ p) or 67(w/al) classes, seq_length.
            """
            base_prob = torch.zeros((codon_prob.size(0),(codon_prob.size(1)%10),codon_prob.size(1)))
            # base_prob: batch_size, 5(w/o p) or 6(w/ p) or 7(w/al) classes, seq_length.
            # PAD, A, C, G, T
            base_index = torch.from_numpy(to_PAD_BASE_GAP_INDEX(position)[:codon_prob.size(1)]).to(device)
            # inputによって変えなければならないので、to_PAD_BASE_GAP_INDEX使うのはどうなんだろう。alignmentに対応できない。
            base_prob = codon_prob.index_add(1,base_index,codon_prob,alpha=1)
            
            return base_prob

        loss = 0
        for i in range(3):
            loss += F.cross_entropy(to_base_prob(input,i+1,self.device),to_base(target,i+1).to(self.device))*self.fst[i]
        return loss

class ols_CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        device,
        epsilon:float = 0.0,
        num_classes:int = 66, #65 pad codon ,# 66 pad codon msk ,# 67 pad codon gap mask
        use_start_codon:bool = False):
        super().__init__()

        self.device = device
        self.matrics = codon_prob_matrics(num_tokens=num_classes,epsilon=epsilon,use_start_codon=use_start_codon)
    def forward(self, input, target):
        target = self.matrics[target] # matricsのサイズは出力クラス^2でなければならない
        target = target.permute(0,2,1)
        target = target.to(self.device)
        loss = F.cross_entropy(input, target)
        return loss