import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, T5ForConditionalGeneration
import math
import numpy as np

class BrainTranslator(nn.Module):
    """BrainTranslator model for EEG-to-Text decoding (CSCL compatible).
    Accepts a pre-trained encoder (e.g., from CSCL) and a seq2seq model.
    Args:
        pre_encoder: Pre-encoder module (can be CSCL-pretrained)
        pretrained_seq2seq: Pretrained sequence-to-sequence model (e.g., BART)
    """
    def __init__(self, pre_encoder, pretrained_seq2seq):
        super(BrainTranslator, self).__init__()
        self.pre_encoder = pre_encoder
        self.seq2seq = pretrained_seq2seq

    def forward(self, src, mask_pre_encoder, mask_seq2seq, labels):
        """
        Args:
            src (Tensor): Word-level EEG (batch_size, seq_len, input_dim)
            mask_pre_encoder (Tensor): Input masks for pre-encoder (1 is masked, 0 is not)
            mask_seq2seq (Tensor): Input masks for seq2seq (0 is masked, 1 is not)
            labels (Tensor): Target labels
        Returns:
            out (Tensor): Output from seq2seq model
        """
        out = self.pre_encoder(src, mask_pre_encoder)
        out = self.seq2seq(
            inputs_embeds=out,
            attention_mask=mask_seq2seq,
            return_dict=True,
            labels=labels
        )
        return out

    def encode(self, src, mask_pre_encoder):
        """
        EEG encoder only (for CSCL training).
        Args:
            src (Tensor): EEG input
            mask_pre_encoder (Tensor): EEG mask
        Returns:
            Encoded features
        """
        return self.pre_encoder(src, mask_pre_encoder)

class BrainTranslatorPreEncoder(nn.Module):
    """Pre-encoder module for BrainTranslator (compatible with CSCL).

    Args:
    ----
        input_dim (int): Dimension of input eeg (default: 840)
        num_layers (int): Number of layers in the pre-encoder (default: 6)
        nhead (int): Number of heads in multiheadattention (default: 8)
        dim_pre_encoder (int): Pre-encoder hidden dimension (default: 2048)
        dim_s2s (int): The seq2seq model hidden dimension (default: 1024)
        dropout (float): Dropout rate (default: 0.1)
    """
    def __init__(self,
                 input_dim=840,
                 num_layers=6,
                 nhead=8,
                 dim_pre_encoder=2048,
                 dim_s2s=1024,
                 dropout=0.1):
        super(BrainTranslatorPreEncoder, self).__init__()
        self.pre_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=dim_pre_encoder,
            dropout=dropout,
            batch_first=True,
            #norm_first=True
        )
        self.pre_encoder_transformer = nn.TransformerEncoder(
            self.pre_encoder_layer,
            num_layers=num_layers
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=input_dim, out_features=dim_s2s, bias=True)
        )

    def forward(self, src, mask_pre_encoder):
        """
        Args:
            src (Tensor): Word-level EEG (batch_size, seq_len, input_dim)
            mask_pre_encoder (Tensor): Input masks (1 is masked, 0 is not)
        Returns:
            out (Tensor): The encoded EEG features (batch_size, seq_len, dim_s2s)
        """
        out = self.pre_encoder_transformer(
            src, src_key_padding_mask=mask_pre_encoder
        )
        out = F.relu(self.fc(out))
        return out


# Example usage for CSCL and downstream task
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_dim = 840
#     num_layers = 6
#     nhead = 8
#     dim_pre_encoder = 2048
#     dim_s2s = 1024

#     # Instantiate pre-encoder
#     pre_encoder = BrainTranslatorPreEncoder(
#         input_dim=input_dim,
#         num_layers=num_layers,
#         nhead=nhead,
#         dim_pre_encoder=dim_pre_encoder,
#         dim_s2s=dim_s2s
#     ).to(device)

#     # Instantiate seq2seq model (BART)
#     config = BartConfig.from_pretrained('facebook/bart-large')
#     s2s = BartForConditionalGeneration.from_pretrained('facebook/bart-large').to(device)

#     # Full BrainTranslator
#     model = BrainTranslator(pre_encoder, s2s).to(device)

#     # Dummy data for testing
#     batch_size = 4
#     seq_len = 56
#     src = torch.rand(batch_size, seq_len, input_dim).to(device)
#     mask_pre_encoder = torch.zeros(batch_size, seq_len, dtype=torch.bool).to(device)  # no mask
#     mask_seq2seq = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)       # all tokens attend
#     labels = torch.ones(batch_size, seq_len, dtype=torch.long).to(device)

#     # Forward pass
#     out = model(src, mask_pre_encoder, mask_seq2seq, labels)
#     print(out.logits.shape)

# """ main architecture for open vocabulary EEG-To-Text decoding"""
# class BrainTranslator(nn.Module):
#     def __init__(self, pretrained_layers, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
#         super(BrainTranslator, self).__init__()
        
#         self.pretrained = pretrained_layers
#         # additional transformer encoder, following BART paper about 
#         self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
#         self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        
#         # print('[INFO]adding positional embedding')
#         # self.positional_embedding = PositionalEncoding(in_feature)

#         self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

#     def addin_forward(self,input_embeddings_batch,  input_masks_invert):
#         """input_embeddings_batch: batch_size*Seq_len*840"""
#         """input_mask: 1 is not masked, 0 is masked"""
#         """input_masks_invert: 1 is masked, 0 is not masked"""

#         # input_embeddings_batch = self.positional_embedding(input_embeddings_batch)
#         # use src_key_padding_masks
#         encoded_embedding = self.additional_encoder(input_embeddings_batch, src_key_padding_mask=input_masks_invert)

#         # encoded_embedding = self.additional_encoder(input_embeddings_batch)
#         encoded_embedding = F.relu(self.fc1(encoded_embedding))
#         return encoded_embedding

#     @torch.no_grad()
#     def generate(
#             self,
#             input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted,
#             generation_config = None,
#             logits_processor = None,
#             stopping_criteria = None,
#             prefix_allowed_tokens_fn= None,
#             synced_gpus= None,
#             assistant_model = None,
#             streamer= None,
#             negative_prompt_ids= None,
#             negative_prompt_attention_mask = None,
#             **kwargs,
#     ):
#         encoded_embedding=self.addin_forward(input_embeddings_batch, input_masks_invert)
#         output=self.pretrained.generate(
#             inputs_embeds = encoded_embedding,
#             attention_mask = input_masks_batch[:,:encoded_embedding.shape[1]],
#             labels = target_ids_batch_converted,
#             return_dict = True,
#             generation_config=generation_config,
#             logits_processor=logits_processor,
#             stopping_criteria=stopping_criteria,
#             prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
#             synced_gpus=synced_gpus,
#             assistant_model=assistant_model,
#             streamer=streamer,
#             negative_prompt_ids=negative_prompt_ids,
#             negative_prompt_attention_mask=negative_prompt_attention_mask,
#             **kwargs,)

#         return output

#     def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
#         encoded_embedding=self.addin_forward(input_embeddings_batch, input_masks_invert)
#         # print(f'forward:{input_embeddings_batch.shape,input_masks_batch.shape,input_masks_invert.shape,target_ids_batch_converted.shape,encoded_embedding.shape}')
#         out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch,
#                               return_dict = True, labels = target_ids_batch_converted)
        
#         return out


from transformers import T5Tokenizer
""" main architecture for open vocabulary EEG-To-Text decoding"""
class T5Translator(nn.Module):
    def __init__(
        self,
        pretrained_model: T5ForConditionalGeneration,
        in_feature: int = 840,
        decoder_embedding_size: int = 1024,
        additional_encoder_nhead: int = 8,
        additional_encoder_dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.pretrained = pretrained_model
        # tokenizer for task prefix
        self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
        # extra EEG transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=in_feature,
            nhead=additional_encoder_nhead,
            dim_feedforward=additional_encoder_dim_feedforward,
            batch_first=True,
        )
        self.additional_encoder = nn.TransformerEncoder(enc_layer, num_layers=6)
        # projection to decoder's embedding size
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def addin_forward(self, input_embeddings: torch.Tensor, input_masks_invert: torch.Tensor) -> torch.Tensor:
        """
        Run the additional EEG encoder and project to decoder embedding size.
        input_embeddings: (batch, seq_len, in_feature)
        input_masks_invert: (batch, seq_len) where 1=masked
        """
        # encode EEG embeddings
        x = self.additional_encoder(input_embeddings, src_key_padding_mask=input_masks_invert)
        # project and activate
        return F.relu(self.fc1(x))

    @torch.no_grad()
    def generate(
        self,
        input_embeddings: torch.Tensor,
        input_masks: torch.Tensor,
        input_masks_invert: torch.Tensor,
        labels: torch.Tensor = None,
        **generate_kwargs
    ):
        """
        Generate sequences from EEG embeddings.
        Assumes generate_kwargs may include:
          - max_length, num_beams, do_sample, repetition_penalty, no_repeat_ngram_size, etc.
        """
        # 1) EEG encoder + projection
        eeg_encoded = self.addin_forward(input_embeddings, input_masks_invert)
        # 2) prepend task prefix to embeddings
        prefix = self.tokenizer("transcribe in English: ", return_tensors="pt").input_ids.to(eeg_encoded.device)
        prefix_emb = self.pretrained.shared(prefix)
        # repeat for batch and concat
        batch_size = eeg_encoded.size(0)
        prefix_emb = prefix_emb.expand(batch_size, -1, -1)
        full_emb = torch.cat([prefix_emb, eeg_encoded], dim=1)
        # extend attention mask
        prefix_mask = torch.ones(batch_size, prefix_emb.size(1), device=eeg_encoded.device)
        full_mask = torch.cat([prefix_mask, input_masks], dim=1)

        # 3) run T5 encoder on full embeddings
        encoder = self.pretrained.get_encoder()
        encoder_outputs = encoder(
            inputs_embeds=full_emb,
            attention_mask=full_mask,
            return_dict=True,
        )

        # 4) call generate with precomputed encoder_outputs
        gen_out = self.pretrained.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=full_mask,
            return_dict_in_generate=True,
            **generate_kwargs,
        )
        return gen_out

    def forward(
        self,
        input_embeddings: torch.Tensor,
        input_masks: torch.Tensor,
        input_masks_invert: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # 1) EEG encoder + projection
        eeg_encoded = self.addin_forward(input_embeddings, input_masks_invert)
        # 2) prepend task prefix
        prefix = self.tokenizer("transcribe in English: ", return_tensors="pt").input_ids.to(eeg_encoded.device)
        prefix_emb = self.pretrained.shared(prefix)
        batch_size = eeg_encoded.size(0)
        prefix_emb = prefix_emb.expand(batch_size, -1, -1)
        full_emb = torch.cat([prefix_emb, eeg_encoded], dim=1)
        prefix_mask = torch.ones(batch_size, prefix_emb.size(1), device=eeg_encoded.device)
        full_mask = torch.cat([prefix_mask, input_masks], dim=1)

        # 3) call T5 model for training
        out = self.pretrained(
            inputs_embeds=full_emb,
            attention_mask=full_mask,
            labels=labels,
            return_dict=True,
        )
        return out


""" crippled open vocabulary EEG-To-Text decoding model w/o additional MTE encoder"""
class BrainTranslatorNaive(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(BrainTranslatorNaive, self).__init__()
        '''no additional transformer encoder version'''
        self.pretrained = pretrained_layers
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def forward(self, input_embeddings_batch, input_masks_batch, input_masks_invert, target_ids_batch_converted):
        """input_embeddings_batch: batch_size*Seq_len*840"""
        """input_mask: 1 is not masked, 0 is masked"""
        """input_masks_invert: 1 is masked, 0 is not masked"""
        encoded_embedding = F.relu(self.fc1(input_embeddings_batch))
        out = self.pretrained(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch, return_dict = True, labels = target_ids_batch_converted)                    
        return out


""" helper modules """
# modified from BertPooler
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('[DEBUG] input size:', x.size())
        # print('[DEBUG] positional embedding size:', self.pe.size())
        x = x + self.pe[:x.size(0), :]
        # print('[DEBUG] output x with pe size:', x.size())
        return self.dropout(x)


""" Miscellaneous (not working well) """
class BrainTranslatorBert(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, hidden_size = 768):
        super(BrainTranslatorBert, self).__init__()

        self.pretrained_Bert = pretrained_layers
        self.fc1 = nn.Linear(in_feature, hidden_size)

    def forward(self, input_embeddings_batch, input_masks_batch, target_ids_batch):
        embedding = F.relu(self.fc1(input_embeddings_batch))
        out = self.pretrained_Bert(inputs_embeds = embedding, attention_mask = input_masks_batch, labels = target_ids_batch, return_dict = True)
        return out

class EEG2BertMapping(nn.Module):
    def __init__(self, in_feature = 840, hidden_size = 512, out_feature = 768):
        super(EEG2BertMapping, self).__init__()
        self.fc1 = nn.Linear(in_feature, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_feature)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out

class ContrastiveBrainTextEncoder(nn.Module):
    def __init__(self, pretrained_text_encoder, in_feature = 840, eeg_encoder_nhead=8, eeg_encoder_dim_feedforward = 2048, embed_dim = 768):
        super(ContrastiveBrainTextEncoder, self).__init__()
        # EEG Encoder
        self.positional_embedding = PositionalEncoding(in_feature)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=eeg_encoder_nhead,  dim_feedforward = eeg_encoder_dim_feedforward, batch_first=True)
        self.EEG_Encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.EEG_pooler = Pooler(in_feature)
        self.ln_final = nn.LayerNorm(in_feature) # to be considered
        
        # project to text embedding
        self.EEG_projection = nn.Parameter(torch.empty(in_feature, embed_dim))
        
        # Text Encoder
        self.TextEncoder = pretrained_text_encoder
        
        # learned temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, input_EEG_features, input_EEG_attn_mask, input_ids, input_text_attention_masks):
        # add positional embedding
        input_EEG_features = self.positional_embedding(input_EEG_features)
        # get EEG feature embedding
        EEG_hiddenstates = self.EEG_Encoder(input_EEG_features,  src_key_padding_mask = input_EEG_attn_mask)
        EEG_hiddenstates = self.ln_final(EEG_hiddenstates)
        EEG_features = self.EEG_pooler(EEG_hiddenstates) # [N, 840]

        # project to text embed size
        EEG_features = EEG_features @ self.EEG_projection # [N, 768]

        # get text feature embedding
        Text_features = self.TextEncoder(input_ids = input_ids, attention_mask = input_text_attention_masks, return_dict = True).pooler_output # [N, 768]
        
        # normalized features
        EEG_features = EEG_features / EEG_features.norm(dim=-1, keepdim=True) # [N, 768]
        Text_features = Text_features / Text_features.norm(dim=-1, keepdim=True) # [N, 768]

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp() 
        logits_per_EEG = logit_scale * EEG_features @ Text_features.t() # [N, N]
        logits_per_text = logit_scale * Text_features @ EEG_features.t() # [N, N]

        return logits_per_EEG, logits_per_text
