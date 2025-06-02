# import os
# import numpy as np
# import h5py
# import data_loading_helpers_modified as dh
# from glob import glob
# from tqdm import tqdm
# import pickle


# task = "NR"

# rootdir = "./dataset/ZuCo/task2-NR-2.0/Matlab_files/"

# print('##############################')
# print(f'start processing ZuCo task2-NR-2.0...')

# dataset_dict = {}

# for file in tqdm(os.listdir(rootdir)):
#     if file.endswith(task+".mat"):

#         file_name = rootdir + file

#         # print('file name:', file_name)
#         subject = file_name.split("ts")[1].split("_")[0]
#         # print('subject: ', subject)

#         # exclude YMH due to incomplete data because of dyslexia
#         if subject != 'YMH':
#             assert subject not in dataset_dict
#             dataset_dict[subject] = []

#             f = h5py.File(file_name,'r')
#             # print('keys in f:', list(f.keys()))
#             sentence_data = f['sentenceData']
#             # print('keys in sentence_data:', list(sentence_data.keys()))
            
#             # sent level eeg 
#             # mean_t1 = np.squeeze(f[sentence_data['mean_t1'][0][0]][()])
#             mean_t1_objs = sentence_data['mean_t1']
#             mean_t2_objs = sentence_data['mean_t2']
#             mean_a1_objs = sentence_data['mean_a1']
#             mean_a2_objs = sentence_data['mean_a2']
#             mean_b1_objs = sentence_data['mean_b1']
#             mean_b2_objs = sentence_data['mean_b2']
#             mean_g1_objs = sentence_data['mean_g1']
#             mean_g2_objs = sentence_data['mean_g2']
            
#             rawData = sentence_data['rawData']
#             contentData = sentence_data['content']
#             # print('contentData shape:', contentData.shape, 'dtype:', contentData.dtype)
#             omissionR = sentence_data['omissionRate']
#             wordData = sentence_data['word']


#             for idx in range(len(rawData)):
#                 # get sentence string
#                 obj_reference_content = contentData[idx][0]
#                 sent_string = dh.load_matlab_string(f[obj_reference_content])
#                 # print('sentence string:', sent_string)
                
#                 sent_obj = {'content':sent_string}
                
#                 # get sentence level EEG
#                 sent_obj['sentence_level_EEG'] = {
#                     'mean_t1':np.squeeze(f[mean_t1_objs[idx][0]][()]), 
#                     'mean_t2':np.squeeze(f[mean_t2_objs[idx][0]][()]), 
#                     'mean_a1':np.squeeze(f[mean_a1_objs[idx][0]][()]), 
#                     'mean_a2':np.squeeze(f[mean_a2_objs[idx][0]][()]), 
#                     'mean_b1':np.squeeze(f[mean_b1_objs[idx][0]][()]), 
#                     'mean_b2':np.squeeze(f[mean_b2_objs[idx][0]][()]), 
#                     'mean_g1':np.squeeze(f[mean_g1_objs[idx][0]][()]), 
#                     'mean_g2':np.squeeze(f[mean_g2_objs[idx][0]][()])
#                 }
#                 # print(sent_obj)
#                 sent_obj['word'] = []

#                 # get word level data
#                 word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = dh.extract_word_level_data(f, f[wordData[idx][0]])
                
#                 if word_data == {}:
#                     print(f'missing sent: subj:{subject} content:{sent_string}, append None')
#                     dataset_dict[subject].append(None)
#                     continue
#                 elif len(word_tokens_all) == 0:
#                     print(f'no word level features: subj:{subject} content:{sent_string}, append None')
#                     dataset_dict[subject].append(None)
#                     continue

#                 else:                    
#                     for widx in range(len(word_data)):
#                         data_dict = word_data[widx]
#                         word_obj = {'content':data_dict['content'], 'nFixations': data_dict['nFix']}
#                         if 'GD_EEG' in data_dict:
#                             # print('has fixation: ', data_dict['content'])
#                             gd = data_dict["GD_EEG"]
#                             ffd = data_dict["FFD_EEG"]
#                             trt = data_dict["TRT_EEG"]
#                             assert len(gd) == len(trt) == len(ffd) == 8
#                             word_obj['word_level_EEG'] = {
#                                 'GD':{'GD_t1':gd[0], 'GD_t2':gd[1], 'GD_a1':gd[2], 'GD_a2':gd[3], 'GD_b1':gd[4], 'GD_b2':gd[5], 'GD_g1':gd[6], 'GD_g2':gd[7]},
#                                 'FFD':{'FFD_t1':ffd[0], 'FFD_t2':ffd[1], 'FFD_a1':ffd[2], 'FFD_a2':ffd[3], 'FFD_b1':ffd[4], 'FFD_b2':ffd[5], 'FFD_g1':ffd[6], 'FFD_g2':ffd[7]},
#                                 'TRT':{'TRT_t1':trt[0], 'TRT_t2':trt[1], 'TRT_a1':trt[2], 'TRT_a2':trt[3], 'TRT_b1':trt[4], 'TRT_b2':trt[5], 'TRT_g1':trt[6], 'TRT_g2':trt[7]}
#                             }
#                             sent_obj['word'].append(word_obj)
                        
#                     sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
#                     sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
#                     sent_obj['word_tokens_all'] = word_tokens_all     
                    
#                     # print(sent_obj.keys())
#                     # print(len(sent_obj['word']))
#                     # print(sent_obj['word'][0])

#                     dataset_dict[subject].append(sent_obj)

# """output"""
# task_name = 'task2-NR-2.0'

# if dataset_dict == {}:
#     print(f'No mat file found for {task_name}')
#     quit()

# output_dir = f'./dataset/ZuCo/{task_name}/pickle'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)

# output_name = f'{task_name}-dataset.pickle'
# # with open(os.path.join(output_dir,'task1-SR-dataset.json'), 'w') as out:
# #     json.dump(dataset_dict,out,indent = 4)

# with open(os.path.join(output_dir,output_name), 'wb') as handle:
#     pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print('write to:', os.path.join(output_dir,output_name))

# """sanity check"""
# print('subjects:', dataset_dict.keys())
# print('num of sent:', len(dataset_dict['YAC']))



import os
import numpy as np
import h5py
import data_loading_helpers_modified as dh
from glob import glob
from tqdm import tqdm
import pickle
import gc
from contextlib import contextmanager

# Memory optimization configuration
BATCH_SIZE = 2  # Process files in batches
FLOAT_DTYPE = np.float32  # Use float32 instead of float64 to halve memory usage
CHUNK_SIZE = 100  # Process sentences in chunks

@contextmanager
def h5py_file_manager(file_path):
    """Context manager for proper file handling"""
    f = None
    try:
        f = h5py.File(file_path, 'r')
        yield f
    finally:
        if f is not None:
            f.close()

def optimize_array_dtype(arr):
    """Convert arrays to memory-efficient dtypes"""
    if arr.dtype == np.float64:
        return arr.astype(FLOAT_DTYPE)
    elif arr.dtype == np.int64:
        return arr.astype(np.int32)
    return arr

def process_sentence_batch(f, sentence_data, start_idx, end_idx, subject):
    """Process a batch of sentences to reduce memory usage"""
    batch_results = []
    
    # Get references for the batch
    rawData = sentence_data['rawData']
    contentData = sentence_data['content']
    wordData = sentence_data['word']
    
    # EEG data references
    eeg_refs = {
        'mean_t1': sentence_data['mean_t1'],
        'mean_t2': sentence_data['mean_t2'],
        'mean_a1': sentence_data['mean_a1'],
        'mean_a2': sentence_data['mean_a2'],
        'mean_b1': sentence_data['mean_b1'],
        'mean_b2': sentence_data['mean_b2'],
        'mean_g1': sentence_data['mean_g1'],
        'mean_g2': sentence_data['mean_g2']
    }
    
    for idx in range(start_idx, min(end_idx, len(rawData))):
        try:
            # Get sentence string
            obj_reference_content = contentData[idx][0]
            sent_string = dh.load_matlab_string(f[obj_reference_content])
            
            sent_obj = {'content': sent_string}
            
            # Get sentence level EEG with memory optimization
            eeg_data = {}
            for key, refs in eeg_refs.items():
                try:
                    data = np.squeeze(f[refs[idx][0]][()])
                    eeg_data[key] = optimize_array_dtype(data)
                except (IndexError, KeyError) as e:
                    print(f"Warning: Missing EEG data {key} for sentence {idx}, subject {subject}")
                    eeg_data[key] = np.array([], dtype=FLOAT_DTYPE)
            
            sent_obj['sentence_level_EEG'] = eeg_data
            sent_obj['word'] = []
            
            # Get word level data
            try:
                word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = \
                    dh.extract_word_level_data(f, f[wordData[idx][0]])
            except (IndexError, KeyError) as e:
                print(f'Error extracting word data: subj:{subject} content:{sent_string}, append None')
                batch_results.append(None)
                continue
            
            if word_data == {}:
                print(f'Missing sent: subj:{subject} content:{sent_string}, append None')
                batch_results.append(None)
                continue
            elif len(word_tokens_all) == 0:
                print(f'No word level features: subj:{subject} content:{sent_string}, append None')
                batch_results.append(None)
                continue
            else:
                for widx in range(len(word_data)):
                    data_dict = word_data[widx]
                    word_obj = {
                        'content': data_dict['content'], 
                        'nFixations': data_dict['nFix']
                    }
                    
                    if 'GD_EEG' in data_dict:
                        gd = optimize_array_dtype(np.array(data_dict["GD_EEG"]))
                        ffd = optimize_array_dtype(np.array(data_dict["FFD_EEG"]))
                        trt = optimize_array_dtype(np.array(data_dict["TRT_EEG"]))
                        
                        if len(gd) == len(trt) == len(ffd) == 8:
                            word_obj['word_level_EEG'] = {
                                'GD': {'GD_t1': gd[0], 'GD_t2': gd[1], 'GD_a1': gd[2], 'GD_a2': gd[3], 
                                      'GD_b1': gd[4], 'GD_b2': gd[5], 'GD_g1': gd[6], 'GD_g2': gd[7]},
                                'FFD': {'FFD_t1': ffd[0], 'FFD_t2': ffd[1], 'FFD_a1': ffd[2], 'FFD_a2': ffd[3], 
                                       'FFD_b1': ffd[4], 'FFD_b2': ffd[5], 'FFD_g1': ffd[6], 'FFD_g2': ffd[7]},
                                'TRT': {'TRT_t1': trt[0], 'TRT_t2': trt[1], 'TRT_a1': trt[2], 'TRT_a2': trt[3], 
                                       'TRT_b1': trt[4], 'TRT_b2': trt[5], 'TRT_g1': trt[6], 'TRT_g2': trt[7]}
                            }
                            sent_obj['word'].append(word_obj)
                
                sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
                sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
                sent_obj['word_tokens_all'] = word_tokens_all
                
                batch_results.append(sent_obj)
                
        except Exception as e:
            print(f"Error processing sentence {idx} for subject {subject}: {str(e)}")
            batch_results.append(None)
    
    return batch_results

def process_single_file(file_path, task):
    """Process a single file with memory optimization"""
    file_name = file_path
    subject = file_name.split("ts")[1].split("_")[0]
    
    # Skip YMH subject
    if subject == 'YMH':
        return subject, []
    
    print(f"Processing subject: {subject}")
    subject_data = []
    
    try:
        with h5py_file_manager(file_name) as f:
            sentence_data = f['sentenceData']
            rawData = sentence_data['rawData']
            total_sentences = len(rawData)
            
            print(f"  Total sentences for {subject}: {total_sentences}")
            
            # Process sentences in chunks to manage memory
            for start_idx in range(0, total_sentences, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, total_sentences)
                print(f"  Processing sentences {start_idx}-{end_idx-1}")
                
                batch_results = process_sentence_batch(f, sentence_data, start_idx, end_idx, subject)
                subject_data.extend(batch_results)
                
                # Force garbage collection after each chunk
                gc.collect()
                
    except Exception as e:
        print(f"Error processing file {file_name}: {str(e)}")
        return subject, []
    
    print(f"  Completed subject {subject}: {len(subject_data)} sentences processed")
    return subject, subject_data

def save_intermediate_results(dataset_dict, output_dir, task_name, batch_num):
    """Save intermediate results to prevent data loss"""
    intermediate_name = f'{task_name}-dataset-batch-{batch_num}.pickle'
    intermediate_path = os.path.join(output_dir, intermediate_name)
    
    with open(intermediate_path, 'wb') as handle:
        pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Intermediate save: {intermediate_path}')

def main():
    task = "NR"
    rootdir = "./dataset/ZuCo/task2-NR-2.0/Matlab_files/"
    
    print('##############################')
    print(f'Start processing ZuCo task2-NR-2.0 (Memory Optimized)...')
    
    # Get all files first
    mat_files = [os.path.join(rootdir, file) for file in os.listdir(rootdir) 
                 if file.endswith(task + ".mat")]
    
    if not mat_files:
        print("No .mat files found!")
        return
    
    print(f"Found {len(mat_files)} files to process")
    
    # Setup output directory
    task_name = 'task2-NR-2.0'
    output_dir = f'./dataset/ZuCo/{task_name}/pickle'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    dataset_dict = {}
    
    # Process files in batches
    for batch_start in range(0, len(mat_files), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(mat_files))
        batch_files = mat_files[batch_start:batch_end]
        batch_num = batch_start // BATCH_SIZE + 1
        
        print(f"\n=== Processing Batch {batch_num} ({len(batch_files)} files) ===")
        
        batch_dict = {}
        
        for file_path in tqdm(batch_files, desc=f"Batch {batch_num}"):
            subject, subject_data = process_single_file(file_path, task)
            if subject_data:  # Only add if data was successfully processed
                batch_dict[subject] = subject_data
            
            # Force garbage collection after each file
            gc.collect()
        
        # Merge batch results into main dataset
        dataset_dict.update(batch_dict)
        
        # Save intermediate results
        save_intermediate_results(dataset_dict, output_dir, task_name, batch_num)
        
        # Clear batch dictionary to free memory
        del batch_dict
        gc.collect()
        
        print(f"Batch {batch_num} completed. Total subjects so far: {len(dataset_dict)}")
    
    # Final save
    if dataset_dict:
        output_name = f'{task_name}-dataset.pickle'
        final_path = os.path.join(output_dir, output_name)
        
        with open(final_path, 'wb') as handle:
            pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'Final save: {final_path}')
        
        # Sanity check
        print('\n=== Final Results ===')
        print('Subjects:', list(dataset_dict.keys()))
        if 'YAC' in dataset_dict:
            print('Num of sentences for YAC:', len(dataset_dict['YAC']))
        print('Total subjects processed:', len(dataset_dict))
    else:
        print('No data was successfully processed!')

if __name__ == "__main__":
    main()
