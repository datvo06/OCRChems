from configure import *
from include import * 

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

from timm.models.vision_transformer import Mlp
from timm.models.layers import DropPath, trunc_normal_

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
