#音声モデルの導入
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Scaled dot-product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class EnhancedConvBlock(nn.Module):
    def __init__(
        self, in_dim, out_dim, kernel_size: int = 3, p_drop: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)
        self.dropout = nn.Dropout(p_drop)
        self.attention = SelfAttention(out_dim, heads=4)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))
        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))
        
        # Apply self-attention
        X = X.permute(0, 2, 1)  # (batch, channels, time) -> (batch, time, channels)
        X = self.attention(X, X, X)
        X = X.permute(0, 2, 1)  # (batch, time, channels) -> (batch, channels, time)
        
        return self.dropout(X)

class BasicConvClassifier(nn.Module):
    def __init__(
        self, num_classes: int, seq_len: int, in_channels: int, hid_dim: int = 128
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            EnhancedConvBlock(in_channels, hid_dim),
            EnhancedConvBlock(hid_dim, hid_dim),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        ###
        #Args:
        #    X (torch.Tensor): Input tensor of shape (b, c, t)
        #Returns:
        #    torch.Tensor: Output tensor of shape (b, num_classes)
        ###
        X = self.blocks(X)
        return self.head(X)





"""
#シンプルなアテンションを導入
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_weights = self.softmax(attn_scores)

        attn_output = torch.matmul(attn_weights, value)
        return attn_output

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128  
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
            ConvBlock(hid_dim, hid_dim),
            ConvBlock(hid_dim, hid_dim),  
            ConvBlock(hid_dim, hid_dim),
        )

        self.attention = SelfAttention(hid_dim)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        ###_summary_
        #Args:
        #    X ( b, c, t ): _description_
        #Returns:
        #    X ( b, num_classes ): _description_
        ###
        
        X = self.blocks(X)
        X = X.permute(0, 2, 1)  # (b, d, t) -> (b, t, d)
        X = self.attention(X)
        X = X.permute(0, 2, 1)  # (b, t, d) -> (b, d, t)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.2, 
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)   

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))  

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)

#以上、シンプルなアテンション
"""





"""（オリジナル）
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128                ##original 128 
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),  ## new 層の追加
            ConvBlock(hid_dim, hid_dim),  ## new 層の追加
            ConvBlock(hid_dim, hid_dim),  ## new 層の追加
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        
        ###_summary_
        #Args:
        #    X ( b, c, t ): _description_
        #Returns:
        #    X ( b, num_classes ): _description_
        ###
        
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.4,               ##変更：過学習が激しいので0.1→0.2→0.4に上げてみた
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)       ##疑問、変更はなし。ちゃんと機能しているの？

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))          ##疑問、変更はなし。ReLUにしたら速くなるか？精度は？

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)
        
    """