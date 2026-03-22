
import torch
import torch.nn as nn
import time
import numpy as np
from numba import njit
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from numba import prange
import os
import datetime
from collections import OrderedDict
import math
from scipy import stats
from kemeny_transformer.data import synthesis as dsyn
from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Optional


###################
# Network definition
# Notation :
#            bsz : batch size
#            numb_items : number of candidates
#            dim_emb : embedding/hidden dimension
#            numb_heads : number of attention heads
#            dim_ff : feed-forward dimension
#            numb_layers : number of encoder/decoder layers
###################
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)


class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self


class EmbeddingType(Enum):
    """Enumeration for the different types of embedding layers available."""
    LINEAR = 'linear'
    CONV = 'conv'

class AbstractEmbedding(nn.Module, ABC):
    """
    Abstract base class for all embedding layers.
    Ensures that any embedding module implements a forward pass and has an embedding_dim attribute.
    """
    def __init__(self):
        super().__init__()
        self.embedding_dim = 0

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes the raw input tensor and returns its embedding.

        Args:
            x (torch.Tensor): The input data. Shape will vary based on the data source.

        Returns:
            torch.Tensor: The embedded data. Shape: (batch_size, seq_len, embedding_dim).
        """
        pass

class LinearEmbedding(AbstractEmbedding):
    """
    A simple linear projection embedding layer, suitable for continuous feature inputs.

    Args:
        input_dim (int): The size of the input feature dimension (e.g., max_voters).
        embedding_dim (int): The desired size of the output embedding.
    """
    def __init__(self, input_dim: int, embedding_dim: int):
        super().__init__()
        self.embedding_layer = nn.Linear(input_dim, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects the input features to the embedding dimension.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, input_dim).

        Returns:
            torch.Tensor: Embedded tensor. Shape: (batch_size, seq_len, embedding_dim).
        """
        return self.embedding_layer(x)

class ConvEmbedding(AbstractEmbedding):
    """
    An embedding layer using 1D convolutions that is robust to a variable number of input features (voters).
    This is useful for capturing local patterns in the voter rankings for each candidate.

    Args:
        embedding_dim (int): The desired size of the output embedding.
        out_channels (int): The number of output channels for the first convolutional layer.
    """
    def __init__(self, embedding_dim: int, out_channels: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels * 2, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        # Instead of AdaptiveAvgPool1d, we'll do masked average manually.
        self.linear = nn.Linear(out_channels * 2, embedding_dim)

    def forward(self, x: torch.Tensor, voter_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Passes the input through the convolutional network to get the embedding.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, variable_input_dim).

        Returns:
            torch.Tensor: Embedded tensor. Shape: (batch_size, seq_len, embedding_dim).
        """
        batch_size, seq_len, input_dim = x.shape
        x_reshaped = x.view(batch_size * seq_len, 1, input_dim)

        if voter_mask is not None:
             if voter_mask.dim() == 2:
                 voter_mask_expanded = voter_mask.unsqueeze(1).expand(batch_size, seq_len, input_dim)
             else:
                 voter_mask_expanded = voter_mask
             voter_mask_batch = voter_mask_expanded.sum(dim=2) # Shape: (batch_size, seq_len)

             # Process each item in the batch independently to avoid padding
             pooled_outs = []
             for i in range(batch_size):
                 # Find the number of valid voters for this item in the batch
                 # All tokens in the same batch item have the same number of valid voters
                 valid_len = voter_mask[i].sum().long().item()
                 # If valid_len is 0 (e.g., padded candidate), just use length 1 to avoid empty tensors
                 if valid_len == 0:
                     valid_len = 1

                 # Extract the unpadded sequence for all candidates in this batch item: shape (seq_len, 1, valid_len)
                 x_item = x[i:i+1, :, :valid_len].view(seq_len, 1, valid_len)

                 # Run the convolutions on the valid segment only for all candidates in this batch item
                 out_item = self.conv1(x_item)
                 out_item = self.relu1(out_item)
                 out_item = self.conv2(out_item)
                 out_item = self.relu2(out_item)

                 # Pool over the valid sequence
                 # The average is calculated securely without any padding tokens
                 pooled_item = out_item.mean(dim=2) # Shape: (seq_len, out_channels*2)
                 pooled_outs.append(pooled_item)

             pooled_out = torch.cat(pooled_outs, dim=0) # Shape: (batch_size * seq_len, out_channels*2)

        else:
             # Fast path if no mask is provided
             out = self.conv1(x_reshaped)
             out = self.relu1(out)
             out = self.conv2(out)
             out = self.relu2(out)
             pooled_out = out.mean(dim=2)

        embedded_x = self.linear(pooled_out)

        # Reshape back to the expected (batch_size, seq_len, embedding_dim)
        return embedded_x.view(batch_size, seq_len, self.embedding_dim)




class Transformer_encoder(nn.Module):
    """
    Encoder network based on self-attention transformer
    Inputs :
      h of size      (bsz, numb_items, dim_emb)    batch of input candidates
      src_key_padding_mask of size (bsz, numb_items) batch of masks for padding
    Outputs :
      h of size      (bsz, numb_items, dim_emb)    batch of encoded candidates
      weights of size  (bsz, numb_heads, numb_items, numb_items) batch of attention scores
      src_key_padding_mask of size (bsz, numb_items) batch of masks for padding (returned for convenience)
    """

    def __init__(self, numb_layers, dim_emb, numb_heads, dim_ff, batchnorm: bool):
        super(Transformer_encoder, self).__init__()
        assert dim_emb % numb_heads == 0, "dim_emb must be divisible by numb_heads"
        self.MHA_layers = nn.ModuleList([nn.MultiheadAttention(dim_emb, numb_heads) for _ in range(numb_layers)])
        self.linear1_layers = nn.ModuleList([nn.Linear(dim_emb, dim_ff) for _ in range(numb_layers)])
        self.linear2_layers = nn.ModuleList([nn.Linear(dim_ff, dim_emb) for _ in range(numb_layers)])
        if batchnorm:
            self.norm1_layers = nn.ModuleList([nn.BatchNorm1d(dim_emb) for _ in range(numb_layers)])
            self.norm2_layers = nn.ModuleList([nn.BatchNorm1d(dim_emb) for _ in range(numb_layers)])
        else:
            self.norm1_layers = nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(numb_layers)])
            self.norm2_layers = nn.ModuleList([nn.LayerNorm(dim_emb) for _ in range(numb_layers)])

        self.numb_layers = numb_layers
        self.numb_heads = numb_heads
        self.batchnorm = batchnorm

    def forward(self, h: torch.Tensor, src_key_padding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the encoder.

        Args:
            h (torch.Tensor): Input tensor. Shape: (bsz, seq_len, dim_emb).
            src_key_padding_mask (torch.Tensor): Padding mask. Shape: (bsz, seq_len).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The encoded tensor. Shape: (batch_size, seq_len, dim_emb).
                - The attention weights. Shape: (batch_size, seq_len, seq_len).
                - The input padding mask. Shape: (batch_size, seq_len).
        """
        # PyTorch nn.MultiheadAttention requires input size (seq_len, bsz, dim_emb)
        h = h.transpose(0, 1)  # size(h)=(seq_len, bsz, dim_emb)
        weights = None # To store weights from the last layer

        for i in range(self.numb_layers):
            h_skip_connection = h
            h, weights = self.MHA_layers[i](h, h, h, key_padding_mask=src_key_padding_mask)
            # add skip connection
            h = h_skip_connection + h
            if self.batchnorm:
                # Pytorch nn.BatchNorm1d requires input size (bsz, dim, seq_len)
                h = h.permute(1, 2, 0).contiguous()
                h = self.norm1_layers[i](h)
                h = h.permute(2, 0, 1).contiguous()
            else:
                h = self.norm1_layers[i](h)

            #feed forward
            h_skip_connection = h
            h = self.linear2_layers[i](torch.relu(self.linear1_layers[i](h)))
            h = h_skip_connection + h
            if self.batchnorm:
                h = h.permute(1, 2, 0).contiguous()
                h = self.norm2_layers[i](h)
                h = h.permute(2, 0, 1).contiguous()
            else:
                h = self.norm2_layers[i](h)  # size(h)=(seq_len, bsz, dim_emb)

        h = h.transpose(0, 1)
        return h, weights, src_key_padding_mask


def newMHA(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
           numb_heads: int, mask: Optional[torch.Tensor] = None, clip_value: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute multi-head attention (MHA) given a query Q, key K, value V and attention mask :
      h = Concat_{k=1}^numb_heads softmax(Q_k^T.K_k).V_k
    Note : We did not use nn.MultiheadAttention to avoid re-computing all linear transformations at each call.

    Inputs :
      Q of size (bsz, seq_len_q, dim_emb)           batch of queries
      K of size (bsz, seq_len_kv, dim_emb)          batch of keys
      V of size (bsz, seq_len_kv, dim_emb)          batch of values
      mask of size (bsz, seq_len_kv)                batch of masks of visited cities
      clip_value is a scalar
    Outputs :
      attn_output of size (bsz, seq_len_q, dim_emb) batch of attention vectors
      attn_weights of size (bsz, seq_len_q, seq_len_kv) batch of attention weights
    """

    bsz, numb_items, emd_dim = K.size()
    seq_len_q = Q.size(1)

    # Reshape Q, K, V for multi-head computation
    if numb_heads > 1:
        Q = Q.transpose(1, 2).contiguous().view(bsz * numb_heads, emd_dim // numb_heads, seq_len_q).transpose(1, 2)
        K = K.transpose(1, 2).contiguous().view(bsz * numb_heads, emd_dim // numb_heads, numb_items).transpose(1, 2)
        V = V.transpose(1, 2).contiguous().view(bsz * numb_heads, emd_dim // numb_heads, numb_items).transpose(1, 2)

    # Calculate attention scores
    attn_weights = torch.bmm(Q, K.transpose(1, 2)) / (Q.size(-1) ** 0.5)

    if clip_value is not None:
        attn_weights = clip_value * torch.tanh(attn_weights)

    if mask is not None:
        if numb_heads > 1:
            mask = torch.repeat_interleave(mask, repeats=numb_heads, dim=0)
        # Apply mask by setting masked positions to negative infinity
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(1), float('-1e9'))

    attn_weights = torch.softmax(attn_weights, dim=-1)
    attn_output = torch.bmm(attn_weights, V)

    if numb_heads > 1:
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, emd_dim, seq_len_q).transpose(1, 2)
        attn_weights = attn_weights.view(bsz, numb_heads, seq_len_q, numb_items).mean(dim=1)

    return attn_output, attn_weights


class AutoRegressiveDecoderLayer(nn.Module):
    """
    Single decoder layer based on self-attention and query-attention
    Inputs :
      h_t of size     (bsz, 1, dim_emb)             batch of input queries
      K_att of size   (bsz, numb_items, dim_emb) batch of query-attention keys
      V_att of size   (bsz, numb_items, dim_emb) batch of query-attention values
      mask of size    (bsz, numb_items)        batch of masks of visited cities
    Output :
      h_t of size (bsz, dim_emb)                    batch of transformed queries
    """

    def __init__(self, dim_emb, numb_heads):
        super(AutoRegressiveDecoderLayer, self).__init__()
        self.dim_emb = dim_emb
        self.numb_heads = numb_heads
        # Self-attention layers
        self.Wq_selfatt = nn.Linear(dim_emb, dim_emb)
        self.Wk_selfatt = nn.Linear(dim_emb, dim_emb)
        self.Wv_selfatt = nn.Linear(dim_emb, dim_emb)
        self.W0_selfatt = nn.Linear(dim_emb, dim_emb)
        # Cross-attention layers
        self.W0_att = nn.Linear(dim_emb, dim_emb)
        self.Wq_att = nn.Linear(dim_emb, dim_emb)
        # Feed-forward layers
        self.W1_MLP = nn.Linear(dim_emb, dim_emb)
        self.W2_MLP = nn.Linear(dim_emb, dim_emb)
        # Normalization layers
        self.BN_selfatt = nn.LayerNorm(dim_emb)
        self.BN_att = nn.LayerNorm(dim_emb)
        self.BN_MLP = nn.LayerNorm(dim_emb)
        # Cached keys and values for self-attention
        self.K_selfatt = None
        self.V_selfatt = None

    def reset_selfatt_keys_values(self):
        """Resets the cached self-attention keys and values."""
        self.K_selfatt = None
        self.V_selfatt = None

    def forward(self, h_t: torch.Tensor, K_att: torch.Tensor, V_att: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for the autoregressive decoder layer."""
        bsz = h_t.size(0)
        # Ensure h_t is (bsz, 1, dim_emb) for sequence processing
        h_t = h_t.view(bsz, 1, self.dim_emb)

        # 1. Masked Self-Attention over the decoded sequence
        q_selfatt = self.Wq_selfatt(h_t)
        k_selfatt = self.Wk_selfatt(h_t)
        v_selfatt = self.Wv_selfatt(h_t)

        # Concatenate the new self-attention key and value to the previous keys and values
        if self.K_selfatt is None:
            self.K_selfatt = k_selfatt
            self.V_selfatt = v_selfatt
        else:
            self.K_selfatt = torch.cat([self.K_selfatt, k_selfatt], dim=1)
            self.V_selfatt = torch.cat([self.V_selfatt, v_selfatt], dim=1)

        # Compute self-attention between candidates in the partial ranking
        h_t_res = h_t
        # No mask needed for self-attention as we attend to all previously decoded items
        h_t_att, _ = newMHA(q_selfatt, self.K_selfatt, self.V_selfatt, self.numb_heads)
        h_t = h_t_res + self.W0_selfatt(h_t_att) #size(h_t)=(bsz,1,dim_emb)
        h_t = self.BN_selfatt(h_t.squeeze(1)).view(bsz, 1, self.dim_emb)

        # 2. Cross-Attention over encoder outputs
        h_t_res = h_t
        q_a = self.Wq_att(h_t)  # size(q_a)=(bsz, 1, bem_dim)
        # Use the provided mask to ignore padded/ranked candidates from the encoder
        h_t_att, _ = newMHA(q_a, K_att, V_att, self.numb_heads, mask)
        h_t = h_t_res + self.W0_att(h_t_att)  # size(h_t)=(bsz, 1, dim_emb)
        h_t = self.BN_att(h_t.squeeze(1)).view(bsz, 1, self.dim_emb)  # size(h_t)=(bsz, 1, dim_emb)

        # 3. MLP (Feed-forward)
        h_t_res = h_t
        h_t_mlp = self.W2_MLP(torch.relu(self.W1_MLP(h_t)))
        h_t = h_t_res + h_t_mlp
        h_t = self.BN_MLP(h_t.squeeze(1))  # size(h_t)=(bsz, dim_emb)
        return h_t


class Transformer_decoder(nn.Module):
    """
    Decoder network based on self-attention and query-attention transformers
    Inputs :
        h_t of size     (bsz, dim_emb)                         batch of input queries
        K_att of size   (bsz, numb_items, dim_emb*numb_layers_decoder) batch of query-attention keys for all decoding layers
        V_att of size   (bsz, numb_items, dim_emb*numb_layers_decoder) batch of query-attention values for all decoding layers
        mask of size    (bsz, numb_items)                 batch of masks of already ranked candidates
    Output :
        prob_next_candidates of size (bsz, numb_items)    batch of probabilities of next node
    """

    def __init__(self, dim_emb, numb_heads, numb_layers_decoder):
        super(Transformer_decoder, self).__init__()
        self.dim_emb = dim_emb
        self.numb_heads = numb_heads
        self.numb_layers_decoder = numb_layers_decoder
        if numb_layers_decoder < 1:
            raise ValueError("numb_layers_decoder must be at least 1.")

        # N-1 standard decoder layers
        self.decoder_layers = nn.ModuleList(
            [AutoRegressiveDecoderLayer(dim_emb, numb_heads) for _ in range(numb_layers_decoder - 1)])
        # Final layer to project query for attention-based probability calculation
        self.Wq_final = nn.Linear(dim_emb, dim_emb)

    # Reset to None self-attention keys and values when decoding starts
    def reset_selfatt_keys_values(self):
        """Resets the cached self-attention keys and values for all layers."""
        for l in range(self.numb_layers_decoder - 1):
            self.decoder_layers[l].reset_selfatt_keys_values()

    def forward(self, h_t: torch.Tensor, K_att: torch.Tensor, V_att: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Transformer decoder stack."""
        # Pass through N-1 standard decoder layers
        for l in range(self.numb_layers_decoder - 1):
            # Get the K, V slices for this layer
            K_att_l = K_att[:, :, l * self.dim_emb:(l + 1) * self.dim_emb].contiguous()
            V_att_l = V_att[:, :, l * self.dim_emb:(l + 1) * self.dim_emb].contiguous()
            h_t = self.decoder_layers[l](h_t, K_att_l, V_att_l, mask)

        # Final (Nth) layer: single-head attention for probability output
        l = self.numb_layers_decoder - 1 # Get index for the final K/V slice
        K_att_l = K_att[:, :, l * self.dim_emb:(l + 1) * self.dim_emb].contiguous()
        V_att_l = V_att[:, :, l * self.dim_emb:(l + 1) * self.dim_emb].contiguous()

        q_final = self.Wq_final(h_t)
        bsz = h_t.size(0)
        q_final = q_final.view(bsz, 1, self.dim_emb)

        # Get attention weights from the final layer. Use 1 head.
        _, att_weights = newMHA(q_final, K_att_l, V_att_l, 1, mask, clip_value=10.0)
        prob_next_node = att_weights.squeeze(1) # Shape: (bsz, numb_items)
        return prob_next_node


def generate_positional_encodeing(d_model, max_len):
    """
    Create standard transformer PEs.
    Inputs :
      d_model is a scalar correspoding to the hidden dimension
      max_len is the maximum length of the sequence
    Output :
      pe of size (max_len, d_model), where d_model=dim_emb, max_len=1000
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class kemeny_transformer(nn.Module):
    """Model
    The kemeny_transformer network is composed of two steps :
      Step 1. Encoder step : Take a set of (voters,numb_items) base rankings
                             and encode the set with self-transformer.
      Step 2. Decoder step : Build the kemeny-optimal ranking recursively/autoregressively,
                             i.e. one kandidate at a time, with a self-transformer and query-transformer.
    Inputs :
      x of size (bsz, numb_items, dim_input_candidates) ranking position of candidates on base rankings
      deterministic is a boolean : If True the kemeny-optimal ranking will choose the city with the highest probability.
                                   If False the salesman will choose the city with Bernouilli sampling.
    Outputs :
      ranking of size (bsz, candidates) : batch of final ranking, i.e. sequences of ordered candidate
                                          ranking [b,t] contains the idx of the candidate picked at step t in batch b
      sumLogProbOfActions of size (bsz,) : batch of sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
    """

    def __init__(self,
                 embedding_type: EmbeddingType,
                 input_dim: int,
                 embedding_dim: int,
                 dim_ff: int,
                 numb_heads: int,
                 numb_layers_decoder: int,
                 numb_layers_encoder:int,
                 max_len_PE: int,
                 conv_out_channels: int = 64,
                 batchnorm: bool = True):
        super(kemeny_transformer, self).__init__()
        self.embedding_type=embedding_type
        self.dim_emb = embedding_dim
        self.input_dim=input_dim
        self.dim_ff=dim_ff

        # --- 1. Embedding Layer (Instantiated based on type) ---
        if embedding_type == EmbeddingType.LINEAR:
            # For LinearEmbedding, input_dim is required and fixed.
            self.embedding_layer = LinearEmbedding(input_dim=input_dim, embedding_dim=embedding_dim)
            print(f"linear embedding model with input dimension {input_dim}, embedding dimension {embedding_dim}")
        elif embedding_type == EmbeddingType.CONV:


            # For ConvEmbedding, input_dim is not needed for initialization,
            # as it can handle variable input lengths.
            self.embedding_layer = ConvEmbedding(embedding_dim=embedding_dim, out_channels=conv_out_channels)
            print(f"conv embedding model with embedding dimension {embedding_dim}, out channels {conv_out_channels}")
        else:
            raise ValueError(f"Unknown embedding_type: '{embedding_type}'. Please use the EmbeddingType enum.")
        print(embedding_type)
        # --- 2. Encoder ---
        self.encoder = Transformer_encoder(numb_layers_encoder, embedding_dim, numb_heads, dim_ff, batchnorm)

        # --- 3. Decoder Placeholder ---
        # vector to start decoding
        self.start_placeholder = nn.Parameter(torch.randn(embedding_dim))


        # --- 4. Decoder ---
        self.decoder = Transformer_decoder(embedding_dim, numb_heads, numb_layers_decoder)

        # --- 5. Decoder K/V Projections ---
        # Linear layers to project encoder outputs into keys and values for decoder's cross-attention.
        self.WK_att_decoder = nn.Linear(embedding_dim, numb_layers_decoder * embedding_dim)
        self.WV_att_decoder = nn.Linear(embedding_dim, numb_layers_decoder * embedding_dim)

        # --- 6. Positional Encoding ---
        self.PE = generate_positional_encodeing(embedding_dim, max_len_PE)

    def forward(self,  x: torch.Tensor, padding_mask: torch.Tensor, voter_mask: torch.Tensor = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the KemenyTransformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_candidates, dim_input_candidates),
                              representing the base rankings.
            padding_mask (torch.Tensor): A boolean mask for the input tensor `x`.
                                         Shape (batch_size, num_candidates). `True` indicates a padded element.
            deterministic (bool): If True, uses argmax for selection. If False, samples
                                  from the output distribution.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - final_rankings (torch.Tensor): The generated rankings, shape (batch_size, num_candidates).
                                             Contains 1-indexed candidate IDs (1 to num_candidates). Padded steps are 0.
            - avg_log_prob_per_step (torch.Tensor): The average log probability for valid (non-padded)
                                                    actions taken, shape (batch_size,).
            - output_mask (torch.Tensor): A boolean mask for the `final_rankings`.
                                          Shape (batch_size, num_candidates). `True` indicates a padded step.
        """

        bsz = x.shape[0]
        numb_candidates = x.shape[1]
        device = x.device
        zero_to_bsz = torch.arange(bsz, device=device)
        # 1. Input Embedding
        # Project input features to embedding dimension.
        # Shape: (batch_size, num_candidates, dim_emb)
        if hasattr(self.embedding_layer, 'forward') and 'voter_mask' in self.embedding_layer.forward.__code__.co_varnames:
            h = self.embedding_layer(x, voter_mask=voter_mask)
        else:
            h = self.embedding_layer(x)  # size(h)=(bsz,numb_candidates,dim_emb)
        # 2. Prepare for Encoder
        # Append the start placeholder to the candidate embeddings. This placeholder
        # will be used to kick off the decoding process.
        # Shape: (batch_size, num_candidates + 1, dim_emb)
        h = torch.cat([h, self.start_placeholder.repeat(bsz, 1, 1)],
                      dim=1)

        # Prepare padding mask for the encoder.
        # The mask is extended for the start_placeholder token, which is never padded.
        # Shape: (batch_size, num_candidates + 1)
        encoder_padding_mask = torch.cat(
            [padding_mask, torch.zeros(bsz, 1, device=device, dtype=torch.bool)],
            dim=1
        )

        # 3. Encoder Forward Pass
        # Create contextualized embeddings. The padding mask ensures attention ignores padded candidates.
        # Shape: (batch_size, num_candidates + 1, dim_emb)
        h_encoder, _, _ = self.encoder(h, src_key_padding_mask=encoder_padding_mask)

        # 4. Prepare for Decoder
        # list that will contain Long tensors of shape (bsz,) that gives the idx of the candidates chosen at time t
        final_ranking_steps = []
        # list that will contain Float tensors of shape (bsz,) that gives the neg log probs of the choices made at time t
        sumLogProbOfActions = []
        # list that will contain boolean masks for the output steps
        output_mask_steps = []

        # Project encoder outputs to K and V for all decoder layers
        K_att_decoder = self.WK_att_decoder(h_encoder)  # size(K_att)=(bsz, num_candidates+1, dim_emb*numb_layers_decoder)
        V_att_decoder = self.WV_att_decoder(h_encoder)  # size(V_att)=(bsz, num_candidates+1, dim_emb*numb_layers_decoder)

        # Get the embedding of the start_placeholder from the encoder output
        self.PE = self.PE.to(device)
        # idx_start_placeholder is the index of the start token, which is `numb_candidates`
        idx_start_placeholder_val = numb_candidates
        idx_start_placeholder = torch.tensor([idx_start_placeholder_val], device=device).long().repeat(bsz)

        h_start = h_encoder[zero_to_bsz, idx_start_placeholder, :] + self.PE[0].repeat(bsz, 1)  # size(h_start)=(bsz, dim_emb)

        # Initialize mask for ranked candidates. Start by masking the placeholder.
        mask_ranked_candidates = torch.zeros(bsz, numb_candidates + 1, device=device).bool()  # False
        mask_ranked_candidates[zero_to_bsz, idx_start_placeholder] = True
        # clear key and value stored in the decoder
        self.decoder.reset_selfatt_keys_values()

        # 5. Autoregressive Decoding Loop
        h_t = h_start

        # Get the number of valid (non-padded) items for each batch item.
        # We subtract 1 from the sum of the original padding_mask
        # because the mask is True for *padded* items.
        # Shape: (bsz,)
        num_valid_items = (~padding_mask).float().sum(dim=1)

        for t in range(numb_candidates):

            # Combine masks: an item is invalid if it's already been ranked OR if it's padding.
            attention_mask = mask_ranked_candidates | encoder_padding_mask


            # Get probabilities for the next candidate
            pro_next_candidate = self.decoder(h_t, K_att_decoder, V_att_decoder,
                                              attention_mask)  # size(prob_next_node)=(bsz, num_candidates+1)

            # Choose next candidate (argmax or sampling)
            if deterministic:
                idx = torch.argmax(pro_next_candidate, dim=1)
            else:
                idx = torch.distributions.Categorical(pro_next_candidate).sample() # Shape: (bsz,)

            # compute logprobs of the action items in the list sumLogProbOfActions
            ProbOfChoices = pro_next_candidate[zero_to_bsz, idx]
            sumLogProbOfActions.append(torch.log(ProbOfChoices))

            # update embedding of the current visited node
            h_t = h_encoder[zero_to_bsz, idx, :]
            h_t = h_t + self.PE[t + 1].expand(bsz, self.dim_emb)

            # is_padded_step is True if t >= num_valid_items for that batch item
            is_padded_step = (t >= num_valid_items) # Shape: (bsz,)

            # Store 0 if padded, otherwise store (idx + 1)
            # We use torch.where to do this efficiently on the batch.
            current_step_ranking = torch.where(
                is_padded_step,
                torch.tensor(0, device=device, dtype=torch.long), # Use 0 for padding
                idx + 1 # Use 1-indexed ID for valid candidates
            )
            final_ranking_steps.append(current_step_ranking)

            # update masks with ranked candidates
            mask_ranked_candidates[zero_to_bsz, idx] = True

        # logprob_of_choices = sum_t log prob( pi_t | pi_(t-1),...,pi_0 )
        # Stack all log probabilities: Shape (bsz, num_candidates)
        all_log_probs = torch.stack(sumLogProbOfActions, dim=1)

        # convert the list of candidates in to a tensor of shape (bsz, num_candidates)
        final_rankings = torch.stack(final_ranking_steps, dim=1)

        # Create the final output mask
        # We can derive this directly from the time step logic
        t_range = torch.arange(numb_candidates, device=device).expand(bsz, numb_candidates)
        output_mask = (t_range >= num_valid_items.unsqueeze(1)) # Shape: (bsz, num_candidates)

        # Mask out log-probs from padded steps
        all_log_probs.masked_fill_(output_mask, 0.0)

        # Sum log-probs for valid steps
        sum_log_prob = all_log_probs.sum(dim=1) # Shape: (bsz,)

        # num_valid_steps is just num_valid_items
        num_valid_steps = num_valid_items # Shape: (bsz,)

        # Calculate average, handle division by zero if a batch item has 0 valid steps
        avg_log_prob_per_step = sum_log_prob / num_valid_steps.clamp(min=1.0)

        return final_rankings, avg_log_prob_per_step, output_mask
