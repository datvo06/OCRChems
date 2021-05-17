import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel
from torch.nn.utils.rnn import *

def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
#     return seed

import os
from datetime import datetime

import math
import numpy as np
import random
import PIL
import cv2
import matplotlib

# std libs
import collections
from collections import defaultdict
import copy
import numbers
import inspect
import shutil
from timeit import default_timer as timer
import itertools
from collections import OrderedDict
from multiprocessing import Pool
import multiprocessing as mp

import json
import zipfile
from shutil import copyfile

import csv
import pandas as pd
import pickle
import glob
import sys
from distutils.dir_util import copy_tree
import time
import itertools as it

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import Levenshtein

PI  = np.pi
INF = np.inf
EPS = 1e-12

def seed_py(seed):
    random.seed(seed)
    np.random.seed(seed)
#     return seed
CUDA_LAUNCH_BLOCKING=1

# -----------------------------------------------------------------------
# CONFIGURE                                                             
# -----------------------------------------------------------------------
STOI = {
    '<sos>': 190,
    '<eos>': 191,
    '<pad>': 192,
}

#---

patch_size   = 16
pixel_pad    = 3
pixel_stride = 4
num_pixel    = (patch_size // pixel_stride)**2
pixel_scale  = 0.8  #1.0  #0.62=36/58 #1.0


#---

vocab_size = 193
max_length = 300 #278 #275


#---

pixel_dim  = 24
patch_dim  = 384

text_dim    = 384
decoder_dim = 384
num_layer = 3
num_head  = 8
ff_dim = 1024

# patch_data_dir = '../input/bms-patch16-scale08/train_patch16_s0.800' # metadata
patch_data_dir = '/ext_data2/comvis/khanhdtq/bms/test_patch16_s0.800/test_patch16_s0.800'
data_dir = '/home/khanhdtq/kaggle/code/hengck23_bms/data' # patch data

fold = 3
out_dir = '/ext_data2/comvis/khanhdtq/bms/tnt_patch16_s0.8/fold%d' % fold
os.makedirs(out_dir, exist_ok=True)
initial_checkpoint = out_dir + '/checkpoint/00922000_model.pth' 

# -----------------------------------------------------------------------
# FAIRSEQ MODEL                                                            
# -----------------------------------------------------------------------
from typing import Tuple, Dict

from fairseq import utils
from fairseq.models import *
from fairseq.modules import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath, trunc_normal_

#https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute
class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

# ------------------------------------------------------
# https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
# https://stackoverflow.com/questions/46452020/sinusoidal-embedding-attention-is-all-you-need

class PositionEncode1D(nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        assert (dim % 2 == 0)
        self.max_length = max_length

        d = torch.exp(torch.arange(0., dim, 2)* (-math.log(10000.0) / dim))
        position = torch.arange(0., max_length).unsqueeze(1)
        pos = torch.zeros(1, max_length, dim)
        pos[0, :, 0::2] = torch.sin(position * d)
        pos[0, :, 1::2] = torch.cos(position * d)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size, T, dim = x.shape
        x = x + self.pos[:,:T]
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

# https://fairseq.readthedocs.io/en/latest/tutorial_simple_lstm.html
# see https://gitlab.maastrichtuniversity.nl/dsri-examples/dsri-pytorch-workspace/-/blob/c8a88cdeb8e1a0f3a2ccd3c6119f43743cbb01e9/examples/transformer/fairseq/models/transformer.py
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
    def forward_one(self,
            x   : Tensor,
            mem : Tensor,
            incremental_state : Optional[Dict[str, Dict[str, Optional[Tensor]]]]
    )-> Tensor:
        x = x[-1:]
        for layer in self.layer:
            x = layer(x, mem, incremental_state=incremental_state)[0]
        x = self.layer_norm(x)
        return x
class Attention(nn.Module):
    """ Multi-Head Attention
    """

    def __init__(self, dim, hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

    def forward(self,
        x: Tensor,
        mask: Optional[Tensor] = None
    )-> Tensor:

        B, N, C = x.shape
        qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = qk[0], qk[1]  # make torchscript happy (cannot use tensor as tuple)
        v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        #---
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
        x = self.proj_drop(x)
        return x




class Block(nn.Module):
    def __init__(self, dim, in_dim, num_pixel, num_heads=12, in_num_head=4, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        # Inner transformer
        self.norm_in = norm_layer(in_dim)
        self.attn_in = Attention(
            in_dim, in_dim, num_heads=in_num_head, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.norm_mlp_in = norm_layer(in_dim)
        self.mlp_in = Mlp(in_features=in_dim, hidden_features=int(in_dim * 4),
                          out_features=in_dim, act_layer=act_layer, drop=drop)

        self.norm1_proj = norm_layer(in_dim)
        self.proj = nn.Linear(in_dim * num_pixel, dim, bias=True)
        # Outer transformer
        self.norm_out = norm_layer(dim)
        self.attn_out = Attention(
            dim, dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm_mlp = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
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

#---------------------------------

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


#---------------------------------



class TNT(nn.Module):
    """ Transformer in Transformer - https://arxiv.org/abs/2103.00112
    """

    def __init__(self,
            patch_size=patch_size,
            embed_dim =patch_dim,
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

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.pixel_embed = PixelEmbed( patch_size=patch_size, in_dim=in_dim, stride=first_stride)
        #num_patches = self.pixel_embed.num_patches
        #self.num_patches = num_patches
        new_patch_size = 4 #self.pixel_embed.new_patch_size
        num_pixel = new_patch_size ** 2

        self.norm1_proj = norm_layer(num_pixel * in_dim)
        self.proj = nn.Linear(num_pixel * in_dim, embed_dim)
        self.norm2_proj = norm_layer(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_pos = nn.Embedding(100*100,embed_dim) #nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pixel_pos = nn.Parameter(torch.zeros(1, in_dim, new_patch_size, new_patch_size))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        for i in range(depth):
            blocks.append(Block(
                dim=embed_dim, in_dim=in_dim, num_pixel=num_pixel, num_heads=num_heads, in_num_head=in_num_head,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer))
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm_layer(embed_dim)

        trunc_normal_(self.cls_token, std=.02)
        #trunc_normal_(self.patch_pos, std=.02)
        trunc_normal_(self.pixel_pos, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'patch_pos', 'pixel_pos', 'cls_token'}


    def forward(self,  patch, coord, mask):
        B = len(patch)
        batch_size, max_of_num_patch, s, s = patch.shape

        patch = patch.reshape(batch_size*max_of_num_patch, 1, s, s).repeat(1,3,1,1)
        pixel_embed = self.pixel_embed(patch, self.pixel_pos)

        patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, max_of_num_patch, -1))))

        #patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
        #patch_embed = patch_embed + self.patch_pos
        #patch_embed[:, 1:] = patch_embed[:, 1:] + self.patch_pos(coord[:, :, 0] * 100 + coord[:, :, 1])

        patch_embed[:,:1]= self.cls_token.expand(B, -1, -1)
        patch_embed= patch_embed + self.patch_pos(coord[:, :, 0] * 100 + coord[:, :, 1])
        patch_embed = self.pos_drop(patch_embed)

        for blk in self.blocks:
            pixel_embed, patch_embed = blk(pixel_embed, patch_embed, mask)

        patch_embed = self.norm(patch_embed)
        return patch_embed

class Net(nn.Module):

    def __init__(self,):
        super(Net, self).__init__()
        self.cnn = TNT()
        self.image_encode = nn.Identity()

        #---
        self.text_pos    = PositionEncode1D(text_dim,max_length)
        self.token_embed = nn.Embedding(vocab_size, text_dim)
        self.text_decode = TransformerDecode(decoder_dim, ff_dim, num_head, num_layer)

        #---
        self.logit  = nn.Linear(decoder_dim, vocab_size)
        self.dropout = nn.Dropout(p=0.5)

        #----
        # initialization
        self.token_embed.weight.data.uniform_(-0.1, 0.1)
        self.logit.bias.data.fill_(0)
        self.logit.weight.data.uniform_(-0.1, 0.1)


    @torch.jit.unused
    def forward(self, patch, coord, token, patch_pad_mask, token_pad_mask):
        device = patch.device
        batch_size = len(patch)
        #---
        patch = patch*2-1
        image_embed = self.cnn(patch, coord, patch_pad_mask)
        image_embed = self.image_encode(image_embed).permute(1,0,2).contiguous()

        text_embed = self.token_embed(token)
        text_embed = self.text_pos(text_embed).permute(1,0,2).contiguous()

        max_of_length = token_pad_mask.shape[-1]
        text_mask = np.triu(np.ones((max_of_length, max_of_length)), k=1).astype(np.uint8)
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask)==1).to(device)

        #----
        # <todo> perturb mask as aug
        text_pad_mask = token_pad_mask[:,:,0]==0
        image_pad_mask = patch_pad_mask[:,:,0]==0
        x = self.text_decode(text_embed[:max_of_length], image_embed, text_mask, text_pad_mask, image_pad_mask)
        x = x.permute(1,0,2).contiguous()
        l = self.logit(x)

        logit = torch.zeros((batch_size, max_length, vocab_size),device=device)
        logit[:,:max_of_length]=l
        return logit

    #submit function has not been coded. i will leave it as an exercise for the kaggler
    @torch.jit.export
    def forward_argmax_decode(self, patch, coord, mask):
    
        image_dim   = 384
        text_dim    = 384
        decoder_dim = 384
        num_layer = 3
        num_head  = 8
        ff_dim    = 1024
    
        STOI = {
            '<sos>': 190,
            '<eos>': 191,
            '<pad>': 192,
        }
        max_length = 300 #278 # 275
    
    
        #---------------------------------
        device = patch.device
        batch_size = len(patch)
    
        patch = patch*2-1
        image_embed = self.cnn(patch, coord, mask)
        image_embed = self.image_encode(image_embed).permute(1,0,2).contiguous()
    
        token = torch.full((batch_size, max_length), STOI['<pad>'],dtype=torch.long, device=device)
        text_pos = self.text_pos.pos
        token[:,0] = STOI['<sos>']
    
    
        #-------------------------------------
        eos = STOI['<eos>']
        pad = STOI['<pad>']

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

# -----------------------------------------------------------------------
# UTILS                                                            
# -----------------------------------------------------------------------

import shutil

import builtins
import re

def normalize_inchi(inchi):
    try:
        mol = Chem.MolFromInchi(inchi)
        if mol is not None:
            try:
                inchi = Chem.MolToInchi(mol)
            except:
                pass
    except:
        pass
    return inchi

class YNakamaTokenizer(object):

    def __init__(self, is_load=True):
        self.stoi = {}
        self.itos = {}

        if is_load:
            self.stoi = read_pickle_from_file(data_dir+'/tokenizer.stoi.pickle')
            self.itos = {k: v for v, k in self.stoi.items()}

    def __len__(self):
        return len(self.stoi)

    def build_vocab(self, text):
        vocab = set()
        for t in text:
            vocab.update(t.split(' '))
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        for i, s in enumerate(vocab):
            self.stoi[s] = i
        self.itos = {k: v for v, k in self.stoi.items()}

    def one_text_to_sequence(self, text):
        sequence = []
        sequence.append(self.stoi['<sos>'])
        for s in text.split(' '):
            sequence.append(self.stoi[s])
        sequence.append(self.stoi['<eos>'])
        return sequence

    def one_sequence_to_text(self, sequence):
        return ''.join(list(map(lambda i: self.itos[i], sequence)))

    def one_predict_to_inchi(self, predict):
        inchi = 'InChI=1S/'
        for p in predict:
            if p == self.stoi['<eos>'] or p == self.stoi['<pad>']:
                break
            inchi += self.itos[p]
        return inchi

    # ---
    def text_to_sequence(self, text):
        sequence = [
            self.one_text_to_sequence(t)
            for t in text
        ]
        return sequence

    def sequence_to_text(self, sequence):
        text = [
            self.one_sequence_to_text(s)
            for s in sequence
        ]
        return text

    def predict_to_inchi(self, predict):
        inchi = [
            self.one_predict_to_inchi(p)
            for p in predict
        ]
        return inchi

def compute_lb_score(predict, truth):
    score = []
    for p, t in zip(predict, truth):
        s = Levenshtein.distance(p, t)
        score.append(s)
    score = np.array(score)
    return score

class Struct(object):
    def __init__(self, is_copy=False, **kwargs):
        self.add(is_copy, **kwargs)

    def add(self, is_copy=False, **kwargs):
        #self.__dict__.update(kwargs)

        if is_copy == False:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, copy.deepcopy(value))
                    #setattr(self, key, value.copy())
                except Exception:
                    setattr(self, key, value)

    def drop(self,  missing=None, **kwargs):
        drop_value = []
        for key, value in kwargs.items():
            try:
                delattr(self, key)
                drop_value.append(value)
            except:
                drop_value.append(missing)
        return drop_value

    def __str__(self):
        text =''
        for k,v in self.__dict__.items():
            text += '\t%s : %s\n'%(k, str(v))
        return text



# log ------------------------------------
def remove_comments(lines, token='#'):
    """ Generator. Strips comments and whitespace from input lines.
    """

    l = []
    for line in lines:
        s = line.split(token, 1)[0].strip()
        if s != '':
            l.append(s)
    return l


def open(file, mode=None, encoding=None):
    if mode == None: mode = 'r'

    if '/' in file:
        if 'w' or 'a' in mode:
            dir = os.path.dirname(file)
            if not os.path.isdir(dir):  os.makedirs(dir)

    f = builtins.open(file, mode=mode, encoding=encoding)
    return f


def remove(file):
    if os.path.exists(file): os.remove(file)


def empty(dir):
    if os.path.isdir(dir):
        shutil.rmtree(dir, ignore_errors=True)
    else:
        os.makedirs(dir)


# http://stackoverflow.com/questions/34950201/pycharm-print-end-r-statement-not-working
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

# io ------------------------------------
def write_list_to_file(list_file, strings):
    with open(list_file, 'w') as f:
        for s in strings:
            f.write('%s\n'%str(s))
    pass


def read_list_from_file(list_file, comment='#'):
    with open(list_file) as f:
        lines  = f.readlines()
    strings=[]
    for line in lines:
        if comment is not None:
            s = line.split(comment, 1)[0].strip()
        else:
            s = line.strip()
        if s != '':
            strings.append(s)
    return strings

def read_pickle_from_file(pickle_file):
    with open(pickle_file,'rb') as f:
        x = pickle.load(f)
    return x

def write_pickle_to_file(pickle_file, x):
    with open(pickle_file, 'wb') as f:
        pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

# etc ------------------------------------
def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError


def np_float32_to_uint8(x, scale=255):
    return (x*scale).astype(np.uint8)

def np_uint8_to_float32(x, scale=255):
    return (x/scale).astype(np.float32)


def int_tuple(x):
    return tuple( [int(round(xx)) for xx in x] )


def df_loc_by_list(df, key, values):
    df = df.loc[df[key].isin(values)]
    df = df.assign(sort = pd.Categorical(df[key], categories=values, ordered=True))
    df = df.sort_values('sort')
    #df = df.reset_index()
    df = df.drop('sort', axis=1)
    return  df

# -----------------------------------------------------------------------
# PATCH DATASET                                                             
# -----------------------------------------------------------------------

import io

# tools to make patches ...
# see https://www.kaggle.com/yasufuminakama/inchi-resnet-lstm-with-attention-inference/data
def remove_rotate(image, orientation):
    l = orientation
    if l == 1:
        image = np.rot90(image,-1)
    if l == 2:
        image = np.rot90(image, 1)
    if l == 3:
        image = np.rot90(image, 2)
    return image


def resize_image(image, scale=1):
    if scale==1 :
        f = pixel_scale * 58/36  #1.2414 #0.80555
        b = int(round(36*0.5))

    if scale==2 :
        f = pixel_scale * 1
        b = int(round(58*0.5))

    image = image[b:-b,b:-b] #remove border
    if not np.isclose(1,f, rtol=1e-02, atol=1e-02):
        h, w = image.shape
        fw = int(round(f*w))
        fh = int(round(f*h))
        image = cv2.resize(image, dsize=(fw, fh), interpolation=cv2.INTER_AREA)
    return image


def repad_image(image, multiplier=16):
    h, w = image.shape
    fh = int(np.ceil(h/multiplier))*multiplier
    fw = int(np.ceil(w/multiplier))*multiplier
    m  = np.full((fh, fw), 255, np.uint8)
    m[0:h, 0:w] = image
    return m
 

def image_to_patch(image, patch_size, pixel_pad, threshold=0):
    p = pixel_pad
    h, w = image.shape

    x, y = np.meshgrid(np.arange(w // patch_size), np.arange(h // patch_size))
    yx = np.stack([y, x], 2).reshape(-1, 2)

    s = patch_size + 2*p
    m = torch.from_numpy(image).reshape(1, 1, h, w).float()
    k = F.unfold(m, kernel_size=s, stride=patch_size, padding=p)
    k = k.permute(0, 2, 1).reshape(-1, s * s)
    k = k.data.cpu().numpy().reshape(-1, s, s)
    #print(k.shape)

    sum = (1 - k[:, p:-p, p:-p]/255).reshape(len(k), -1).sum(-1)
    i = np.where(sum > threshold)
    #print(sum)
    patch = k[i]
    coord = yx[i]
    return  patch, coord


def patch_to_image(patch, coord, width, height):
    image = np.full((height,width), 255, np.uint8)
    p = pixel_pad
    patch = patch[:, p:-p, p:-p]
    num_patch = len(patch)

    for i in range(num_patch):
        y,x = coord[i]
        x = x * patch_size
        y = y * patch_size
        image[y:y+patch_size,x:x+patch_size] = patch[i]
        cv2.rectangle(image, (x, y), (x + patch_size, y + patch_size), 128, 1)
    return image

#<todo>
# np compression is very slow!!!
# https://stackoverflow.com/questions/39035983/compress-zip-numpy-arrays-in-memory
def compress_array(k):
    compressed_k = io.BytesIO()
    np.savez_compressed(compressed_k, k)
    return compressed_k

def uncompress_array(compressed_k):
    compressed_k.seek(0)
    k  = np.load(compressed_k,allow_pickle=True)['arr_0']
    return k

#<todo> add token to mark 4 corner of image
#############################################################################################
def make_chessbord_image(w, h, patch_size=16):
    m = np.zeros((h, w), np.float32)
    s = patch_size
    u = 1
    for y in range(0, h, s):
        v = u
        for x in range(0, w, s):
            m[y:y + s, x:x + s] = v
            v *= -1
        u *= -1
    m = ((0.5 * m + 0.5) * 255).astype(np.uint8)
    return m

def make_fold(mode='train-1'):
    if 'train' in mode:
        df = read_pickle_from_file(data_dir+'/df_train.more.csv.pickle')
        df_fold = pd.read_csv(data_dir+'/df_fold.csv')
        # df_fold = pd.read_csv(data_dir+'/df_fold.fine.csv')
        df_meta = pd.read_csv(data_dir+'/df_train_image_meta.csv')
        df = df.merge(df_fold, on='image_id')
        df = df.merge(df_meta, on='image_id')
        df.loc[:,'path']='train_patch16_s0.800'

        df['fold'] = df['fold'].astype(int)
        #print(df.groupby(['fold']).size()) #404_031
        #print(df.columns)

        fold = int(mode[-1])
        df_train = df[df.fold != fold].reset_index(drop=True)
        df_valid = df[df.fold == fold].reset_index(drop=True)
        return df_train, df_valid

    # Index(['image_id', 'InChI'], dtype='object')
    if 'test' in mode:
        df = pd.read_csv(data_dir+'/sample_submission.csv')
        # df = pd.read_csv(data_dir+'/submit_lb3.80.csv')
        df_meta = pd.read_csv(data_dir+'/df_test_image_meta.csv')
        df = df.merge(df_meta, on='image_id')

        df.loc[:, 'path'] = 'test'
        #df.loc[:, 'InChI'] = '0'
        df.loc[:, 'formula'] = '0'
        df.loc[:, 'text'] =  '0'
        df.loc[:, 'sequence'] = pd.Series([[0]] * len(df))
        df.loc[:, 'length'] = df.InChI.str.len()

#         df_test = df
        
        df_test = df[(df['image_id'] != '381ecfd9c1ff') & (df['image_id'] != 'b45da6cd9f45')].reset_index()
        
        return df_test

# tokenization, padding, ...
def pad_sequence_to_max_length(sequence, max_length, padding_value):
    batch_size =len(sequence)
    pad_sequence = np.full((batch_size,max_length), padding_value, np.int32)
    for b, s in enumerate(sequence):
        L = len(s)
        pad_sequence[b, :L, ...] = s
    return pad_sequence

def load_tokenizer():
    tokenizer = YNakamaTokenizer(is_load=True)
    print('len(tokenizer) : vocab_size', len(tokenizer))
    for k,v in STOI.items():
        assert  tokenizer.stoi[k]==v
    return tokenizer

def null_augment(r):
    return r

class BmsDataset(Dataset):
    def __init__(self, df, tokenizer, augment=null_augment):
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.augment = augment
        self.length = len(self.df)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\tdf  = %s\n'%str(self.df.shape)

        g = self.df['length'].values.astype(np.int32)//20
        g = np.bincount(g, minlength=14)
        string += '\tlength distribution\n'
        for n in range(14):
            string += '\t\t %3d = %8d (%0.4f)\n'%((n+1)*20,g[n], g[n]/g.sum() )
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        token = d.sequence

        patch_file = os.path.join(patch_data_dir, '/train_patch16_s0.800/%s/%s/%s/%s.pickle'%(d.image_id[0], 
                                                            d.image_id[1], d.image_id[2], d.image_id))
        k = read_pickle_from_file(patch_file)

        patch = uncompress_array(k['patch'])
        patch = np.concatenate([
            np.zeros((1, patch_size+2*pixel_pad, patch_size+2*pixel_pad), np.uint8),
            patch],0) #cls token

        coord  = k['coord']
        w = k['width' ]
        h = k['height']

        h = h // patch_size -1
        w = w // patch_size -1
        coord = np.insert(coord, 0, [h, w], 0) #cls token

        r = {
            'index'    : index,
            'image_id' : d.image_id,
            'InChI'    : d.InChI,
            'd' : d,
            'token' : token,
            'patch' : patch,
            'coord' : coord,
        }
        if self.augment is not None: r = self.augment(r)
        return r

class TestBmsDataset(Dataset):
    def __init__(self, df, tokenizer, mode='valid', augment=null_augment):
        super().__init__()
        self.tokenizer = tokenizer
        self.df = df
        self.mode = mode 
        self.augment = augment
        self.length = len(self.df)

    def __str__(self):
        string  = ''
        string += '\tlen = %d\n'%len(self)
        string += '\tdf  = %s\n'%str(self.df.shape)

        g = self.df['length'].values.astype(np.int32)//20
        g = np.bincount(g, minlength=14)
        string += '\tlength distribution\n'
        for n in range(14):
            string += '\t\t %3d = %8d (%0.4f)\n'%((n+1)*20,g[n], g[n]/g.sum() )
        return string

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        token = d.sequence

        if self.mode == 'test':
            patch_file = os.path.join(patch_data_dir, '%s/%s/%s/%s.pickle'%(d.image_id[0], 
                                                            d.image_id[1], d.image_id[2], d.image_id))
        elif self.mode == 'valid':
            patch_file = os.path.join(patch_data_dir, '%s/%s/%s/%s.pickle'%(d.image_id[0], 
                                                            d.image_id[1], d.image_id[2], d.image_id))            
        k = read_pickle_from_file(patch_file)

        patch = uncompress_array(k['patch'])
        patch = np.concatenate([
            np.zeros((1, patch_size+2*pixel_pad, patch_size+2*pixel_pad), np.uint8),
            patch],0) #cls token

        coord  = k['coord']
        w = k['width' ]
        h = k['height']

        h = h // patch_size -1
        w = w // patch_size -1
        coord = np.insert(coord, 0, [h, w], 0) #cls token

        r = {
            'index'    : index,
            'image_id' : d.image_id,
            'InChI'    : d.InChI,
            'd' : d,
            'token' : token,
            'patch' : patch,
            'coord' : coord,
        }
        if self.augment is not None: r = self.augment(r)
        return r

def null_collate(batch, is_sort_decreasing_length=True):
    collate = defaultdict(list)

    if is_sort_decreasing_length: #sort by decreasing length
        sort  = np.argsort([-len(r['token']) for r in batch])
        batch = [batch[s] for s in sort]

    for r in batch:
        for k, v in r.items():
            collate[k].append(v)
    #----

    batch_size = len(batch)
    collate['length'] = [len(l) for l in collate['token']]

    token  = [np.array(t,np.int32) for t in collate['token']]
    token  = pad_sequence_to_max_length(token, max_length=max_length, padding_value=STOI['<pad>'])
    collate['token'] = torch.from_numpy(token).long()

    max_of_length = max(collate['length'])
    token_pad_mask  = np.zeros((batch_size, max_of_length, max_of_length))
    for b in range(batch_size):
        L = collate['length'][b]
        token_pad_mask [b, :L, :L] = 1 #+1 for cls_token

    collate['token_pad_mask'] = torch.from_numpy(token_pad_mask).byte()

    collate['num_patch'] = [len(l) for l in collate['patch']]

    max_of_num_patch = max(collate['num_patch'])
    patch_pad_mask  = np.zeros((batch_size, max_of_num_patch, max_of_num_patch))
    patch = np.full((batch_size, max_of_num_patch, patch_size+2*pixel_pad, patch_size+2*pixel_pad),255) #pad as 255
    coord = np.zeros((batch_size, max_of_num_patch, 2))
    for b in range(batch_size):
        N = collate['num_patch'][b]
        patch[b, :N] = collate['patch'][b]
        coord[b, :N] = collate['coord'][b]
        patch_pad_mask [b, :N, :N] = 1 #+1 for cls_token

#     collate['patch'] = torch.from_numpy(patch).half() / 255
    collate['patch'] = torch.from_numpy(patch).float() / 255
    collate['coord'] = torch.from_numpy(coord).long()
    collate['patch_pad_mask' ] = torch.from_numpy(patch_pad_mask).byte()
    return collate

is_mixed_precision = False   

import torch.cuda.amp as amp
if is_mixed_precision:
    class AmpNet(Net):
        @torch.cuda.amp.autocast()
        def forward(self, *args):
            return super(AmpNet, self).forward(*args)
else:
    AmpNet = Net

def do_predict(net, tokenizer, test_loader):

    text = []

    start_timer = timer()
    test_num = 0
    for t, batch in enumerate(test_loader):
        
        batch_size = len(batch['index'])
        # image = batch['image'].cuda()
        patch = batch['patch'].cuda()
        coord = batch['coord'].cuda()
        mask = batch['patch_pad_mask'].cuda()
        net.eval()
        with torch.no_grad():
            #k = net(image)
            k = net.forward_argmax_decode(patch, coord, mask)

            k = k.data.cpu().numpy()
            k = tokenizer.predict_to_inchi(k)

            text.extend(k)
        
        test_num += batch_size
        print('\r %8d / %d  %s' % (test_num, len(test_loader.dataset), time_to_str(timer() - start_timer, 'sec')),
              end='', flush=True)
                 
    assert(test_num == len(test_loader.dataset))
    print('')
    return text    

def run_submit():

    gpu_no = int(os.environ['CUDA_VISIBLE_DEVICES'])
    
    abnormal_list = ['381ecfd9c1ff', 'b45da6cd9f45']

#     gpu_no = int(os.environ['CUDA_VISIBLE_DEVICES'])

#     fold = 3
#     out_dir = '/ext_data2/comvis/khanhdtq/bms/tnt_patch16_s0.8/fold%d' % fold
#     initial_checkpoint = out_dir + '/checkpoint/00922000_model.pth'

    is_norm_ichi = True

    ## setup  ----------------------------------------
    mode = 'remote' #'remote'     

    if mode == 'local':
        submit_dir = out_dir + '/valid/%s-%s-gpu%d'%(mode, 
                                initial_checkpoint[-18:-4], gpu_no)
    elif mode == 'remote':
        submit_dir = out_dir + '/test/%s-%s-gpu%d'%(mode, 
                                initial_checkpoint[-18:-4],gpu_no)

    os.makedirs(submit_dir, exist_ok=True)

    log = Logger()
    log.open(out_dir + '/log.submit.txt', mode='a')
    log.write('is_norm_ichi = %s\n' % is_norm_ichi)
    log.write('\n')    

    ## dataset ------------------------------------
    tokenizer = load_tokenizer()
    if mode == 'remote':
        df_test = make_fold('test')
        if gpu_no==2 :  df_test = df_test[ : 808053]
        if gpu_no==3 :  df_test = df_test[ 808053:]

    elif mode == 'local':
        df_train, df_test = make_fold('train-%d' % fold)
        
    df_test = df_test.sort_values('length').reset_index(drop=True)
    test_dataset = TestBmsDataset(df_test, tokenizer,
                            mode='valid' if mode == 'local' else 'test')
    test_dataloader = DataLoader(
        test_dataset,
        sampler = SequentialSampler(test_dataset),
        batch_size  = 64,
        drop_last   = False,
        num_workers = 4,
        pin_memory  = True,  
        collate_fn  = lambda batch: null_collate(batch, False),      
    )
    log.write('mode : %s\n'%(mode))
    log.write('test_dataset : \n%s\n'%(test_dataset))

    ## net ----------------------------------------
    net = Net().cuda() #AmpNet().cuda()
    net.load_state_dict(torch.load(initial_checkpoint)['state_dict'], strict=True)
    net = torch.jit.script(net)

    #---
    start_timer = timer()
    predict = do_predict(net, tokenizer, test_dataloader)
    log.write('time %s \n' % time_to_str(timer() - start_timer, 'min'))

    #----
    if is_norm_ichi:
        predict = [normalize_inchi(t) for t in predict] 
        
    old_submission = pd.read_csv('/home/khanhdtq/kaggle/code/hengck23_bms/results/tnt-s-224-fairseq/fold3/test/submission-00298000_model.csv')
    old_submission = old_submission[old_submission['image_id'].isin(abnormal_list)].reset_index(drop=True)
    
    df_submit = pd.DataFrame()
    df_submit.loc[:,'image_id'] = df_test.image_id.values
    df_submit.loc[:,'InChI'] = predict #
    df_submit = pd.concat([df_submit, old_submission])
    df_submit.to_csv(submit_dir + '/submit.csv', index=False)

    log.write('submit_dir : %s\n' % (submit_dir))
    log.write('initial_checkpoint : %s\n' % (initial_checkpoint))
    log.write('df_submit : %s\n' % str(df_submit.shape))
    log.write('\n')        

    if mode == 'local':
        truth = df_test['InChI'].values.tolist()
        lb_score = compute_lb_score(predict, truth)

        log.write('lb_score  = %f\n'%lb_score.mean())
        log.write('is_norm_ichi = %s\n' % is_norm_ichi)
        log.write('\n')
        
        df_eval = df_submit.copy()
        df_eval.loc[:,'truth']=truth
        df_eval.loc[:,'lb_score']=lb_score
        df_eval.loc[:,'length'] = df_test['length']
        df_eval.to_csv(submit_dir + '/df_eval.csv', index=False)        

if __name__ == '__main__':

    run_submit()

