import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_

class LMFE(nn.Module):
    def __init__(self, dim, reduction=8):
        super(LMFE, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.spp_conv_1 = nn.Sequential(nn.Conv2d(dim, dim//reduction, kernel_size=1, stride=1, padding=0, bias=True),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(dim // reduction, dim // reduction, kernel_size=3, stride=1, padding=1, bias=True),
                                        nn.LeakyReLU(inplace=True))
        self.spp_conv_2 = nn.Sequential(nn.Conv2d(dim, dim//reduction, kernel_size=1, stride=1, padding=0, bias=True),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(dim//reduction, dim//reduction, kernel_size=5, stride=1, padding=2, bias=True),
                                        nn.LeakyReLU(inplace=True))
        self.spp_conv_3 = nn.Sequential(nn.Conv2d(dim, dim//reduction, kernel_size=1, stride=1, padding=0, bias=True),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Conv2d(dim//reduction, dim//reduction, kernel_size=7, stride=1, padding=3, bias=True),
                                        nn.LeakyReLU(inplace=True))

        self.gelu = nn.GELU()
        self.spp_fusion = nn.Conv2d(dim//reduction*3, dim, kernel_size=1, padding=0)
        self.att_conv = nn.Conv2d(dim, dim*4, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)
        self.project = nn.Conv2d(dim*2, dim, kernel_size=1, bias=True)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = recovery_layer(self.norm(trans_layer(x)), h, w)
        short_cut = x
        x_1 = self.spp_conv_1(x)
        x_2 = self.spp_conv_2(x)
        x_3 = self.spp_conv_3(x)
        x = torch.cat([x_1, x_2, x_3], dim=1)
        x = self.spp_fusion(x)
        x = x + short_cut

        short_cut = x
        x1, x2 = self.att_conv(x).chunk(2, dim=1)
        x = self.gelu(x1) * x2
        x = self.project(x)
        x = x + short_cut
        return x

def trans_layer(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def recovery_layer(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class GFE(nn.Module):
    def __init__(self, dim,  num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias=False)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor=2, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = recovery_layer(self.norm1(trans_layer(x)), h, w)
        x = x + self.attn(x)
        x = recovery_layer(self.norm2(trans_layer(x)), h, w)
        x = x + self.ffn(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.relu = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x = self.relu(x)
        x = self.project_out(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.LeakyReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LLIEFormerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.local_forward = LMFE(dim)
        self.global_forward = GFE(dim, num_heads)
        self.fc_fusion = nn.Conv2d(dim * 2, dim, kernel_size=1, padding=0)

    def forward(self, x):
        x_global_text = self.global_forward(x)
        x_local_text = self.local_forward(x)

        x = torch.cat([x_global_text, x_local_text], dim=1)
        x = self.fc_fusion(x)

        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, drop=0., norm_layer=nn.LayerNorm, downsample=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            LLIEFormerBlock(dim=dim,num_heads=num_heads)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = Downsample(dim//2, dim)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)

        for blk in self.blocks:
                x = blk(x)
        return x

class BasicLayer_up(nn.Module):
    def __init__(self, dim, depth, num_heads, drop=0., norm_layer=nn.LayerNorm, upsample=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        # build blocks
        self.blocks = nn.ModuleList([
            LLIEFormerBlock(dim=dim, num_heads=num_heads)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = Upsample(dim, dim//2)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        out = self.conv(x)
        return out


class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        out = self.deconv(x)
        return out

#-------------------------------------------------------------input and output Proj--------------------------------------------#
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

class OutputProj(nn.Module):
    def __init__(self, in_channel=24, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x


class LLIEFormer(nn.Module):
    def __init__(self, in_chans=3,
                 embed_dim=24, depths=[2, 2, 4, 2], depths_decoder=[2, 4, 2, 2], num_heads=[1, 2, 4, 8],
                 drop_rate=0., norm_layer=nn.LayerNorm, use_attention_map=True):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)

        if use_attention_map == True:
            self.in_chans = in_chans + 1
        else:
            self.in_chans = in_chans

        self.input_proj = InputProj(in_channel=self.in_chans, out_channel=embed_dim, kernel_size=3, stride=1,
                                    act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=embed_dim, out_channel=in_chans*2, kernel_size=3, stride=1)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               drop=drop_rate,
                               norm_layer=norm_layer,
                               downsample=Downsample if (i_layer <= self.num_layers - 1 and i_layer > 0) else None)
            self.layers.append(layer)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Conv2d(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      kernel_size=1, padding=0) if i_layer > 0 else nn.Identity()
            layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                     depth=depths_decoder[(self.num_layers - 1 - i_layer)],
                                     num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                     drop=drop_rate,
                                     norm_layer=norm_layer,
                                     upsample=Upsample if (i_layer < self.num_layers - 1) else None)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = norm_layer(self.embed_dim)
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.5)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.2)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.concat((x, 1 - x_max), dim=1)

        x = self.input_proj(x)

        x_downsample = []

        for layer in self.layers:
            x = layer(x)
            x_downsample.append(x)

        return x, x_downsample

    # Decoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], 1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        return x

    def forward(self, x):
        x_origin = x
        H, W = x.shape[2:]
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.output_proj(x)

        K, B = torch.split(x, (3, 3), dim=1)
        x = K * x_origin + B + x_origin
        x = x[:, :, :H, :W]
        return x