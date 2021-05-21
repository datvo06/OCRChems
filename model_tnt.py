import torch.nn as nn
import torch
from fairseq import utils
from fairseq.models import *
from fairseq.modules import *
from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath, trunc_normal_
import math
from utils import CFG
from tokenizer import tokenizer


# ------------------------------------------------------
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
# https://stackoverflow.com/questions/46452020/sinusoidal-embedding-attention-is-all-you-need

class PositionEncode1D(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        assert(dim % 2 == 0)
        self.max_length = max_length

        d = torch.exp(torch.arange(0., dim, 2) * (-math.log(10000.0)/dim))
        position = torch.arange(0., max_length).unsqueeze(1)
        pos = torch.zeros(1, max_length, dim)
        pos[0, :, 0::2] = torch.sin(position*d)
        pos[0, :, 1::2] = torch.cos(position*d)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size, T, dim = x.shape
        x = x + self.pos[:, :T]
        return x

# https://gitlab.maastrichtuniversity.nl/dsri-examples/dsri-pytorch-workspace/-/blob/c8a88cdeb8e1a0f3a2ccd3c6119f43743cbb01e9/examples/transformer/fairseq/models/transformer.py
#https://github.com/pytorch/fairseq/issues/568
# fairseq/fairseq/models/fairseq_encoder.py

# https://github.com/pytorch/fairseq/blob/master/fairseq/modules/transformer_layer.py
class TransformerEncode(FairseqEncoder):

    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__({})
        #print('my TransformerEncode()')

        self.layer = nn.ModuleList([
            TransformerEncoderLayer(Namespace({
                'encoder_embed_dim': dim,
                'encoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'encoder_normalize_before': True,
                'encoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x):# T x B x C
        #print('my TransformerEncode forward()')
        for layer in self.layer:
            x = layer(x)
        x = self.layer_norm(x)
        return x


class TransformerDecode(FairseqIncrementalDecoder):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__({})
        #print('my TransformerDecode()')

        self.layer = nn.ModuleList([
            TransformerDecoderLayer(Namespace({
                'decoder_embed_dim': dim,
                'decoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'decoder_normalize_before': True,
                'decoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)


    def forward(self, x, mem, x_mask, x_pad_mask, mem_pad_mask):
            #print('my TransformerDecode forward()')
            for layer in self.layer:
                x = layer(
                    x,
                    mem,
                    self_attn_mask=x_mask,
                    self_attn_padding_mask=x_pad_mask,
                    encoder_padding_mask=mem_pad_mask,
                )[0]
            x = self.layer_norm(x)
            return x  # T x B x C

    #def forward_one(self, x, mem, incremental_state):
    def forward_one(self, x:Tensor, mem:Tensor,
                    incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    )-> Tensor:
        x = x[-1:]
        for layer in self.layer:
            x = layer(x, mem, incremental_state=incremental_state)[0]
        x = self.layer_norm(x)
        return x


class Attention(nn.Module):
    '''Multi Head Attention'''
    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_dop = nn.Dropout(proj_drop, inplace=True)

    def forward(self, x:Tensor, mask:Optional[Tensor] = None) -> Tensor:
        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
        # 2, B, num_heads, N, head_dim
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale # B x self.num_heads x NxN
        if mask is not None:
            #mask = mask.unsqueeze(1).repeat(1,self.num_heads,1,1)
            mask = mask.unsqueeze(1).expand(-1,self.num_heads,-1,-1)
            attn = attn.masked_fill(mask == 0, -6e4)
            # attn = attn.masked_fill(mask == 0, -half('inf'))
            # https://github.com/NVIDIA/apex/issues/93
            # How to use fp16 training with masked operations

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x == self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, in_dim, num_pixel, num_head=12, in_num_head=4,
                 mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # Inner transformer
        self.norm_in = norm_layer(in_dim)
        self.attn_in = Attention(
            in_dim, in_dim, num_heads=in_num_head, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm_mlp_in = norm_layer(in_dim)
        self.mlp_in = Mlp(in_features=in_dim, hidden_features=int(in_dim*4),
                          out_features=in_dim, act_layer=act_layer,
                          drop=drop)

        self.norm1_proj = norm_layer(in_dim)
        self.proj = nn.Linear(in_dim * num_pixel, dim, bias=True)

        # Outer Transformer
        self.norm_out = norm_layer(dim)
        self.attn_out = Attention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio),
                       out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, pixel_embed, patch_embed, mask):
        # inner
        pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
        pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
        # outer
        B, N, C = patch_embed.size()
        patch_embed[:, 1:] = patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N, -1))[:, 1:]
        patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed), mask))
        patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
        return pixel_embed, patch_embed


class PixelEmbed(nn.Module):

    def __init__(self,  patch_size=16, in_dim=48, stride=4):
        super().__init__()
        self.in_dim = in_dim
        self.proj = nn.Conv2d(3, self.in_dim, kernel_size=7, padding=0, stride=stride)

    def forward(self, patch, pixel_pos):
        BN = len(patch)
        x = patch
        x = self.proj(x)
        #x = x.transpose(1, 2).reshape(B * self.num_patches, self.in_dim, self.new_patch_size, self.new_patch_size)
        x = x + pixel_pos
        x = x.reshape(BN, self.in_dim, -1).transpose(1, 2)
        return x


class TNT(nn.Module):
    """ Transformer in Transformer - https://arxiv.org/abs/2103.00112
    """
    def __init__(self,
                 patch_size=patch_size,
                 embed_dim=patch_dim,
                 in_dim=pixel_dim,
                 depth=12,
                 num_heads=6,
                 in_num_head=4,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 first_stride=pixel_stride):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.pixel_embed = PixelEmbed(patch_size=patch_size,
                                      in_dim=in_dim, stride=first_stride)
        new_patch_size = 4
        num_pixel = new_patch_size ** 2
        self.norm1_proj = norm_layer(num_pixel * in_dim)
        self.proj = nn.Linear(num_pixel * in_dim, embed_dim)
        self.norm2_proj = norm_layer(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_pos = nn.Embedding(100*100, embed_dim)
        self.pixel_pos = nn.Parameter(torch.zeros(1, in_dim,
                                                  new_patch_size,
                                                  new_patch_size))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        for i in range(depth):
            blocks.append(Block(
                dim=embed_dim, in_dim=in_dim, num_pixel=num_pixel,
                num_heads=num_heads, in_num_head=in_num_head,
                drop_path=dpr[i], norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pixel_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(self.pixel_pos, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init_constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'patch_pos', 'pixel_pos', 'cls_token'}

    def forward(self, patch, coord, mask):
        B = len(patch)
        batch_size, max_of_num_patch, s, s = patch.shape

        patch = patch.reshape(batch_size * max_of_num_patch, 1, s, s).repeat(
            1, 3, 1, 1)
        pixel_embed = self.pixel_embed(patch, self.pixel_pos)
        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, max_of_num_patch, -1))))
        patch_embed[::-1] = self.cls_token.expand(B, -1, -1)
        patch_embed = patch_embed + self.patch_pos(coord[:, :, 0] * 100 + coord[:, :, 1])
        patch_embed = self.pos_drop(patch_embed)
        for blk in self.blocks:
            pixel_embed, patch_embed = blk(pixel_embed, patch_embed, mask)

        patch_embed = self.norm(patch_embed)
        return patch_embed

class TnTNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn = TNT()

        self.image_encode = nn.Identity()
        self.text_pos    = PositionEncode1D(text_dim,max_length)
        self.token_embed = nn.Embedding(vocab_size, text_dim)
        self.text_decode = TransformerDecode(decoder_dim, ff_dim, num_head, num_layer)

        self.logit  = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        self.token_embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)

    @torch.jit.unused
    def forward(self, patch, coord, token, patch_pad_mask, token_pad_mask):
        device = patch.device
        batch_size = len(patch)
        #---
        patch = patch*2 - 1
        image_embed = self.cnn(patch, coord, patch_pad_mask)
        image_embed = self.image_encode(image_embed).permute(1, 0, 2).contiguous()

        # Encoding decoded text
        text_embed = self.token_embed(token)
        text_embed = self.text_pos(text_embed).permute(1, 0, 2).contiguous()

        max_of_length = token_pad_mask.shape[-1]
        # Create lower triangle of the mask
        text_mask = np.triu((max_of_length, max_of_length), k=1).astype(np.uint8)
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)==1).to(device)

        # <todo> pertub mask as aug
        text_pad_mask = token_pad_mask[:, :, 0] == 0
        image_pad_mask = patch_pad_mask[:,:,0] == 0
        # Transformer decode
        # x, mem, x_mask, x_pad_mask, mem_pad_mask
        x = self.text_decode(text_embed[:max_of_length], image_embed,
                             text_mask, text_pad_mask, image_pad_mask)
        x = x.permute(1, 0, 2).contiguous()

        logit = torch.zeros((batch_size, max_length, vocab_size), device=device)
        logit[: , :max_of_length]=l
        return logit


    @torch.jit.export
    def forward_argmax_decode(self, patch, coord, mask):

        image_dim   = CFG.size      # 384
        text_dim    = CFG.text_dim  # 384
        decoder_dim = CFG.decoder_dim   # 384
        num_layer = CFG.num_layer       # 3
        num_head  = CFG.num_head        # 8
        ff_dim    = CFG.ff_dim          # 1024

        max_length = 300 #278 # 275


        #---------------------------------
        device = patch.device
        batch_size = len(patch)

        patch = patch*2-1
        image_embed = self.cnn(patch, coord, mask)
        image_embed = self.image_encode(image_embed).permute(1,0,2).contiguous()

        token = torch.full((batch_size, max_length), tokenizer.stoi['<pad>'],dtype=torch.long, device=device)
        text_pos = self.text_pos.pos
        token[:,0] = tokenizer.stoi['<sos>']


        #-------------------------------------
        eos = tokenizer.stoi['<eos>']
        pad = tokenizer.stoi['<pad>']

        # fast version
        if 1:
            #incremental_state = {}
            incremental_state = torch.jit.annotate(
                Dict[str, Dict[str, Optional[Tensor]]],
                torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
            )
            for t in range(max_length-1):
                #last_token = token [:,:(t+1)]
                #text_embed = self.token_embed(last_token)
                #text_embed = self.text_pos(text_embed) #text_embed + text_pos[:,:(t+1)] #

                last_token = token[:, t]
                text_embed = self.token_embed(last_token)
                text_embed = text_embed + text_pos[:,t] #
                text_embed = text_embed.reshape(1,batch_size,text_dim)

                x = self.text_decode.forward_one(text_embed, image_embed, incremental_state)
                x = x.reshape(batch_size,decoder_dim)

                l = self.logit(x)
                k = torch.argmax(l, -1)  # predict max
                token[:, t+1] = k
                if ((k == eos) | (k == pad)).all():  break

        predict = token[:, 1:]
        return predict


# loss #################################################################
def seq_cross_entropy_loss(logit, token, length):
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    return loss

# https://www.aclweb.org/anthology/2020.findings-emnlp.276.pdf
def seq_focal_cross_entropy_loss(logit, token, length):
    gamma = 0.5 # {0.5,1.0}
    #label_smooth = 0.90

    #---
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    #loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    #non_pad = torch.where(truth != STOI['<pad>'])[0]  # & (t!=STOI['<sos>'])


    # ---
    #p = F.softmax(logit,-1)
    #logp = - torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))

    logp = F.log_softmax(logit, -1)
    logp = logp.gather(1, truth.reshape(-1,1)).reshape(-1)
    p = logp.exp()

    loss = - ((1 - p) ** gamma)*logp  #focal
    #loss = - ((1 + p) ** gamma)*logp  #anti-focal
    loss = loss.mean()
    return loss
