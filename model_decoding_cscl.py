# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data
# from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, T5ForConditionalGeneration
# import math
# import numpy as np

# class BrainTranslator(nn.Module):
#     def __init__(self, pre_encoder, pretrained_seq2seq):
#         super().__init__()
#         self.pre_encoder = pre_encoder
#         self.seq2seq     = pretrained_seq2seq

#     def forward(self, src, mask_pre_encoder, mask_seq2seq, labels):
#         out = self.pre_encoder(src, mask_pre_encoder)
#         return self.seq2seq(
#             inputs_embeds  = out,
#             attention_mask = mask_seq2seq,
#             return_dict    = True,
#             labels         = labels
#         )

#     def encode(self, src, mask_pre_encoder):
#         return self.pre_encoder(src, mask_pre_encoder)

#     @torch.no_grad()
#     def generate(self, src, mask_pre_encoder, mask_seq2seq, **generate_kwargs):
#         embeds = self.pre_encoder(src, mask_pre_encoder)
#         return self.seq2seq.generate(
#             inputs_embeds  = embeds,
#             attention_mask = mask_seq2seq,
#             **generate_kwargs
#         )

# class BrainTranslatorPreEncoder(nn.Module):
#     """Pre-encoder module for BrainTranslator (compatible with CSCL).

#     Args:
#     ----
#         input_dim (int): Dimension of input eeg (default: 840)
#         num_layers (int): Number of layers in the pre-encoder (default: 6)
#         nhead (int): Number of heads in multiheadattention (default: 8)
#         dim_pre_encoder (int): Pre-encoder hidden dimension (default: 2048)
#         dim_s2s (int): The seq2seq model hidden dimension (default: 1024)
#         dropout (float): Dropout rate (default: 0.1)
#     """
#     def __init__(self,
#                  input_dim=840,
#                  num_layers=6,
#                  nhead=8,
#                  dim_pre_encoder=2048,
#                  dim_s2s=1024,
#                  dropout=0.1):
#         super(BrainTranslatorPreEncoder, self).__init__()
#         self.pre_encoder_layer = nn.TransformerEncoderLayer(
#             d_model=input_dim,
#             nhead=nhead,
#             dim_feedforward=dim_pre_encoder,
#             dropout=dropout,
#             batch_first=True,
#             #norm_first=True
#         )
#         self.pre_encoder_transformer = nn.TransformerEncoder(
#             self.pre_encoder_layer,
#             num_layers=num_layers
#         )
#         self.fc = nn.Sequential(
#             nn.LayerNorm(input_dim),
#             nn.Dropout(p=dropout),
#             nn.Linear(in_features=input_dim, out_features=dim_s2s, bias=True)
#         )

#     def forward(self, src, mask_pre_encoder):
#         """
#         Args:
#             src (Tensor): Word-level EEG (batch_size, seq_len, input_dim)
#             mask_pre_encoder (Tensor): Input masks (1 is masked, 0 is not)
#         Returns:
#             out (Tensor): The encoded EEG features (batch_size, seq_len, dim_s2s)
#         """
#         out = self.pre_encoder_transformer(
#             src, src_key_padding_mask=mask_pre_encoder
#         )
#         out = F.relu(self.fc(out))
#         return out
