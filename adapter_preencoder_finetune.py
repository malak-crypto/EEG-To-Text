import os
import json
import pickle
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import get_config
from data_cscl import ZuCo_dataset
from model_decoding import BrainTranslatorPreEncoder, BrainTranslator, BrainTranslatorNaive, T5Translator
from transformers import BartForConditionalGeneration, PegasusForConditionalGeneration, T5ForConditionalGeneration, BertGenerationDecoder

def adapter_finetune(args):
    # Load training config
    cfg = json.load(open(args.config, 'r'))
    bands = cfg['eeg_bands']
    # Build dataset list for all tasks
    tasks = cfg['task_name'].split('_')
    whole = []
    if 'task1' in task_name:
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle'
        with open(dataset_path_task3, 'rb') as handle:
            whole.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle' 
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole.append(pickle.load(handle))
    print()
    # Tokenizer
    from transformers import BartTokenizer, PegasusTokenizer, T5Tokenizer
    if args.model == 'BrainTranslator' or args.model=='BrainTranslatorNaive':
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    elif args.model=='PegasusTranslator':
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    else:
        tokenizer = T5Tokenizer.from_pretrained('t5-large')
    # DataLoader
    train_set = ZuCo_dataset(whole, 'train', tokenizer,
                              subject=cfg['subjects'], eeg_type=cfg['eeg_type'], bands=bands,
                              setting='unique_sent', test_input=cfg['train_input'])
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    # Build model
    device = torch.device(args.cuda if torch.cuda.is_available() else 'cpu')
    # Pre-encoder
    pre_encoder = BrainTranslatorPreEncoder(
        input_dim=105*len(bands), num_layers=1, nhead=1,
        dim_pre_encoder=2048, dim_s2s=1024, dropout=0)
    # Decoder
    if args.model=='BrainTranslator' or args.model=='BrainTranslatorNaive':
        decoder = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    elif args.model=='PegasusTranslator':
        decoder = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
    elif args.model=='T5Translator':
        decoder = T5ForConditionalGeneration.from_pretrained('t5-large')
    else:
        decoder = BertGenerationDecoder.from_pretrained('google-bert/bert-large-uncased', is_decoder=True)
    model = BrainTranslator(pre_encoder=pre_encoder, pretrained_seq2seq=decoder)
    # Load checkpoint
    state = torch.load(args.checkpoint, map_location='cpu')
    state = {k.replace('module.', ''):v for k,v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).train()
    # Freeze decoder
    for name, p in model.named_parameters():
        if 'pre_encoder' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # Training loop
    for epoch in range(args.epochs):
        total_loss = 0.0
        for emb, seq_len, mask, mask_inv, tgt_ids, tgt_mask, _ in loader:
            emb, mask, mask_inv, tgt_ids = emb.to(device), mask.to(device), mask_inv.to(device), tgt_ids.to(device)
            # TF forward
            out = model(emb, mask, mask_inv, tgt_ids)
            logits = out.logits  # [B,T,V]
            B,T,V = logits.size()
            loss = F.cross_entropy(logits.view(-1,V), tgt_ids.view(-1), ignore_index=tokenizer.pad_token_id)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch} loss: {total_loss/len(loader):.4f}")
    # Save adapter-tuned model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-checkpoint', required=True)
    parser.add_argument('-config', required=True)
    parser.add_argument('-model', required=True)
    parser.add_argument('-batch_size', type=int, default=8)
    parser.add_argument('-epochs', type=int, default=2)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-cuda', default='cuda:0')
    parser.add_argument('-output', required=True)
    args = parser.parse_args()
    adapter_finetune(args)
