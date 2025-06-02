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

task = "NR"
rootdir = "./dataset/ZuCo/task2-NR-2.0/Matlab_files/"
task_name = 'task2-NR-2.0'
output_dir = f'./dataset/ZuCo/{task_name}/pickle'

print('##############################')
print(f'start processing ZuCo task2-NR-2.0...')

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# Get list of files to process
files = [f for f in os.listdir(rootdir) if f.endswith(task + ".mat")]
batch_size = 3  # Process 3 files at a time to balance memory usage

for i in tqdm(range(0, len(files), desc="Processing batches"):
    batch_files = files[i:i+batch_size]
    batch_data = {}
    
    for file in batch_files:
        file_name = os.path.join(rootdir, file)
        subject = file_name.split("ts")[1].split("_")[0]

        # Skip excluded subject
        if subject == 'YMH':
            continue

        try:
            f = h5py.File(file_name, 'r')
            sentence_data = f['sentenceData']
            
            # Get all needed objects
            mean_objs = {
                'mean_t1': sentence_data['mean_t1'],
                'mean_t2': sentence_data['mean_t2'],
                'mean_a1': sentence_data['mean_a1'],
                'mean_a2': sentence_data['mean_a2'],
                'mean_b1': sentence_data['mean_b1'],
                'mean_b2': sentence_data['mean_b2'],
                'mean_g1': sentence_data['mean_g1'],
                'mean_g2': sentence_data['mean_g2']
            }
            
            rawData = sentence_data['rawData']
            contentData = sentence_data['content']
            wordData = sentence_data['word']
            
            subject_sentences = []
            
            for idx in range(len(rawData)):
                # Get sentence string
                obj_reference_content = contentData[idx][0]
                sent_string = dh.load_matlab_string(f[obj_reference_content])
                
                sent_obj = {'content': sent_string}
                
                # Get sentence level EEG and convert to float32
                sent_obj['sentence_level_EEG'] = {
                    key: np.squeeze(f[mean_objs[key][idx][0]][()]).astype(np.float32)
                    for key in mean_objs
                }
                
                sent_obj['word'] = []
                
                # Get word level data
                word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = dh.extract_word_level_data(
                    f, f[wordData[idx][0]]
                )
                
                if not word_data or len(word_tokens_all) == 0:
                    print(f'No word data: subj:{subject} sent:{sent_string}')
                    subject_sentences.append(None)
                    continue
                    
                for widx in range(len(word_data)):
                    data_dict = word_data[widx]
                    word_obj = {'content': data_dict['content'], 'nFixations': data_dict['nFix']}
                    
                    if 'GD_EEG' in data_dict:
                        # Convert all EEG arrays to float32
                        gd = [x.astype(np.float32) for x in data_dict["GD_EEG"]]
                        ffd = [x.astype(np.float32) for x in data_dict["FFD_EEG"]]
                        trt = [x.astype(np.float32) for x in data_dict["TRT_EEG"]]
                        
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
                subject_sentences.append(sent_obj)
                
            batch_data[subject] = subject_sentences
            f.close()
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            batch_data[subject] = []
        
        # Explicit cleanup
        del f
        gc.collect()
    
    # Save batch results
    for subject, data in batch_data.items():
        subject_file = os.path.join(output_dir, f"{subject}.pkl")
        with open(subject_file, 'wb') as sub_handle:
            pickle.dump(data, sub_handle)
    
    # Clear batch memory
    del batch_data
    gc.collect()

# Combine all subject files into final dataset
final_output = os.path.join(output_dir, f'{task_name}-dataset.pickle')
combined_data = {}

print("Combining subject files...")
for sub_file in tqdm(glob.glob(os.path.join(output_dir, '*.pkl'))):
    try:
        subject = os.path.basename(sub_file).split('.')[0]
        with open(sub_file, 'rb') as f:
            combined_data[subject] = pickle.load(f)
    except Exception as e:
        print(f"Error loading {sub_file}: {str(e)}")

# Save final combined dataset
with open(final_output, 'wb') as handle:
    pickle.dump(combined_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('Final dataset written to:', final_output)

# Sanity check
print('subjects:', combined_data.keys())
if 'YAC' in combined_data:
    print('num of sent for YAC:', len(combined_data['YAC']))
else:
    print('YAC data not found in combined dataset')
