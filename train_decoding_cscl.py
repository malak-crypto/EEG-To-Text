import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, PegasusTokenizer, PegasusForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
from data_cscl import ZuCo_dataset, build_CSCL_maps
from model_decoding import BrainTranslator,BrainTranslatorPreEncoder, BrainTranslatorNaive, T5Translator
from config import get_config
from CSCL import CSCL
import torch.nn.functional as F
import wandb

def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path_best='./checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last='./checkpoints/decoding/last/temp_decoding.pt', dataset_sizes=None, tokenizer=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, subject, sentence in tqdm(dataloaders[phase]):
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)
                    loss = seq2seqLMoutput.loss

                    if phase == 'train':
                        loss.sum().backward()
                        optimizer.step()

                running_loss += loss.sum().item() * input_embeddings_batch.size()[0]

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')

    model.load_state_dict(best_model_wts)
    return model

def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)

# --- CSCL TRAINING LOGIC ---
def train_CSCL(model, dataloaders, cscl, T, optimizer, epochs, device, use_wandb):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())

    for level in range(3):
        best_loss = float('inf')

        for epoch in range(epochs):
            print(f'CSCL Epoch {epoch}/{epochs - 1}, Level {level}')
            print('-' * 10)

            for phase in ['dev', 'train']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                loader = dataloaders[phase]

                for batch, (EEG, _, _, _, _, _, subject, sentence) in enumerate(loader):
                    triplet = cscl[phase].get_triplet(EEG, subject, sentence, level)
                    if triplet is None:
                        print("No Triplets this sample")
                        continue 
                    E, E_pos, E_neg, mask, mask_pos, mask_neg = triplet
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        mask_triplet = torch.vstack((mask, mask_pos, mask_neg)).to(device)
                        out = model(torch.vstack((E, E_pos, E_neg)).to(device), mask_triplet)
                        mask_triplet = abs(mask_triplet-1).unsqueeze(-1)
                        out = (out * mask_triplet).sum(1) / mask_triplet.sum(1)
                        

                        B = E.size(0)
                        h = out[:B]
                        h_pos = out[B:2*B]
                        h_neg = out[2*B:]


                        num = torch.exp(F.cosine_similarity(h, h_pos, dim=1)/T)
                        denom = torch.empty_like(num, device=num.device)
                        for j in range(B):
                            # sum over all positives and negatives
                            denom_j = (torch.exp(F.cosine_similarity(h[j], h_pos, dim=0) / T).sum()
                                       + torch.exp(F.cosine_similarity(h[j], h_neg, dim=0) / T).sum())
                            denom[j] = denom_j
                        loss = -torch.log(num / denom).mean()
                        print(f'{epoch}.{batch} {phase} Loss: {loss.item():.4f}')

                        if phase == 'train':
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()

                        if use_wandb:
                            wandb.log({f"{phase} batch loss": loss.item()})

                        running_loss += loss.item()

                epoch_loss = running_loss / len(loader)
                print(f'{phase} Loss: {epoch_loss:.4f}')

                if use_wandb:
                    wandb.log({f"{phase} epoch loss": epoch_loss})

                if phase == 'dev' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

            print()

    time_elapsed = time.time() - since
    print(f'CSCL Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:4f}')

    model.load_state_dict(best_model_wts)
    return model

# ---- MAIN PIPELINE ----
if __name__ == '__main__':
    args = get_config('train_decoding')
    print(args)

    # --- NEW: CSCL config option ---
    use_cscl_pretraining = args.get('cscl', True)
    cscl_epochs = args.get('cscl_epochs', 5)
    cscl_lr = args.get('cscl_lr', 1e-6)
    cscl_T = args.get('cscl_T', 1)
    cscl_batch_size = args.get('cscl_batch_size',1)
    cscl_wandb = args.get('cscl_wandb', False)

    dataset_setting = 'unique_sent'
    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']
    batch_size = args['batch_size']

    model_name = args['model_name']
    task_name = args['task_name']
    train_input = args['train_input']
    print("train_input is:", train_input)
    save_path = args['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = args['use_random_init']
    device_ids = [0]

    if use_random_init and skip_step_one:
        step2_lr = 5*1e-4

    print(f'[INFO]using model: {model_name}')

    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_{use_cscl_pretraining}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_{use_cscl_pretraining}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}'

    if use_random_init:
        save_name = 'randinit_' + save_name

    save_path_best = os.path.join(save_path, 'best')
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)

    output_checkpoint_name_best = os.path.join(save_path_best, f'{save_name}.pt')

    save_path_last = os.path.join(save_path, 'last')
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)

    output_checkpoint_name_last = os.path.join(save_path_last, f'{save_name}.pt')

    subject_choice = args['subjects']
    eeg_type_choice = args['eeg_type']
    bands_choice = args['eeg_bands']

    print(f'![Debug]using {subject_choice}')
    print(f'[INFO]eeg type {eeg_type_choice}')
    print(f'[INFO]using bands {bands_choice}')

    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if torch.cuda.is_available():
        dev = args['cuda']
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    whole_dataset_dicts = []
    whole_dataset_dicts_task1 = []

    if 'task1' in task_name:
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            data1 = pickle.load(handle)
            whole_dataset_dicts.append(data1)
            whole_dataset_dicts_task1.append(data1)
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle'
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle'
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    print()

    cfg_dir = './config/decoding/'
    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)
    with open(os.path.join(cfg_dir, f'{save_name}.json'), 'w') as out_config:
        json.dump(args, out_config, indent=4)

    # --- Tokenizer selection ---
    if model_name in ['BrainTranslator', 'BrainTranslatorNaive']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    elif model_name == 'PegasusTranslator':
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    elif model_name == 'T5Translator':
        tokenizer = T5Tokenizer.from_pretrained("t5-large")

    # --- DataLoader construction ---
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, test_input=train_input)
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, test_input=train_input)
    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=4)
    val_dataloader = DataLoader(dev_set, batch_size=1, shuffle=False, num_workers=4)
    dataloaders = {'train': train_dataloader, 'dev': val_dataloader}

    # ---- OPTIONAL: CSCL CURRICULUM TRAINING ---- #
    if use_cscl_pretraining:
        print('[INFO] Running CSCL curriculum-aware contrastive pretraining...')
        # CSCL model
        cscl_preencoder = BrainTranslatorPreEncoder(
            input_dim=105*len(bands_choice),  # adjust if needed!
            num_layers=1,  # or from config
            nhead=1,
            dim_pre_encoder=2048,
            dim_s2s=1024,
            dropout=0
        ).to(device)
        # if cscl_wandb:
        #     wandb.init(project='CSCL', reinit=True, config=args)
        #     wandb.watch(cscl_preencoder, log='all')

    # build_CSCL_maps and CSCL expect ZuCo dataset interface!
    train_set_cscl = ZuCo_dataset(whole_dataset_dicts_task1, 'train', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, test_input=train_input)
    dev_set_cscl   = ZuCo_dataset(whole_dataset_dicts_task1, 'dev', tokenizer, subject=subject_choice, eeg_type=eeg_type_choice, bands=bands_choice, setting=dataset_setting, test_input=train_input)
    cscl_train_loader = DataLoader(train_set_cscl, batch_size=cscl_batch_size, shuffle=False, drop_last=False)
    cscl_dev_loader   = DataLoader(dev_set_cscl,   batch_size=cscl_batch_size, shuffle=False, drop_last=False)
    cscl_dataloaders  = {'train': cscl_train_loader, 'dev': cscl_dev_loader}

    fs_train, fp_train, S_train = build_CSCL_maps(train_set_cscl)
    fs_dev, fp_dev, S_dev = build_CSCL_maps(dev_set_cscl)
    cscl_train_obj = CSCL(fs_train, fp_train, S_train)
    cscl_dev_obj = CSCL(fs_dev, fp_dev, S_dev)
    cscl_objs = {'train': cscl_train_obj, 'dev': cscl_dev_obj}

    cscl_optimizer = optim.Adam(params=cscl_preencoder.parameters(), lr=cscl_lr)
    cscl_preencoder = train_CSCL(cscl_preencoder, cscl_dataloaders, cscl_objs, cscl_T, cscl_optimizer, cscl_epochs, device, cscl_wandb)
    print('[INFO] Finished CSCL curriculum pretraining.')

        # Now use the pre-trained encoder weights in your main model!
        # For example, you can pass cscl_preencoder as encoder to BrainTranslator below.

    # ---- Main Model Setup ----
    if model_name == 'BrainTranslator':
        if use_random_init:
            config = BartConfig.from_pretrained('facebook/bart-large')
            pretrained = BartForConditionalGeneration(config)
        else:
            pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        # If using CSCL pretraining, pass the weights!
        if use_cscl_pretraining:
            # cscl_preencoder should be an instance of BrainTranslatorPreEncoder, already trained
            # pretrained is your BART model (e.g., BartForConditionalGeneration)
            model = BrainTranslator(pre_encoder=cscl_preencoder, pretrained_seq2seq=pretrained)
        else:
            # If not using CSCL pretraining, initialize a new pre-encoder
            pre_encoder = BrainTranslatorPreEncoder(
                input_dim=105*len(bands_choice),  # adjust if needed!
                num_layers=1,
                nhead=1,
                dim_pre_encoder=2048,
                dim_s2s=1024,
                dropout=0
            )
            model = BrainTranslator(pre_encoder=pre_encoder, pretrained_seq2seq=pretrained)

    elif model_name == 'BrainTranslatorNaive':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = BrainTranslatorNaive(pretrained, in_feature=105*len(bands_choice), decoder_embedding_size=1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048)

    elif model_name == 'PegasusTranslator':
        pretrained = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
        model = BrainTranslator(pretrained, in_feature=105*len(bands_choice), decoder_embedding_size=1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048)

    elif model_name == 'T5Translator':
        pretrained = T5ForConditionalGeneration.from_pretrained("t5-large")
        model = T5Translator(pretrained, in_feature=105*len(bands_choice), decoder_embedding_size=1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward=2048)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)


    if model_name in ['BrainTranslator', 'BrainTranslatorNaive', 'PegasusTranslator', 'T5Translator']:
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                    continue
                else:
                    param.requires_grad = False

    elif model_name == 'BertGeneration':
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('embeddings' in name) or ('encoder.layer.0' in name):
                    continue
                else:
                    param.requires_grad = False

    # ---- PIPELINE LOGIC: skip step one or not ----
    ######################################################
    # STEP ONE TRAINING: freeze most transformer params
    ######################################################
    if skip_step_one:
        if load_step1_checkpoint:
            stepone_checkpoint = 'path_to_step_1_checkpoint.pt'
            print(f'skip step one, load checkpoint: {stepone_checkpoint}')
            model.load_state_dict(torch.load(stepone_checkpoint))
        else:
            print('skip step one, start from scratch at step two')
    else:
        optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)
        exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        print('=== start Step1 training ... ===')
        show_require_grad_layers(model)
        model = train_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1, checkpoint_path_best=output_checkpoint_name_best, checkpoint_path_last=output_checkpoint_name_last, dataset_sizes=dataset_sizes, tokenizer=tokenizer)

    ######################################################
    # STEP TWO TRAINING: update whole model for a few iterations
    ######################################################
    for name, param in model.named_parameters():
        param.requires_grad = True

    optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)
    exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    print()
    print('=== start Step2 training ... ===')
    show_require_grad_layers(model)

    trained_model = train_model(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, checkpoint_path_best=output_checkpoint_name_best, checkpoint_path_last=output_checkpoint_name_last, dataset_sizes=dataset_sizes, tokenizer=tokenizer)

    # torch.save(trained_model.state_dict(), os.path.join(save_path,output_checkpoint_name))
