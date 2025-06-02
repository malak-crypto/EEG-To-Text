import os
import numpy as np
import h5py
import data_loading_helpers_modified as dh
from glob import glob
from tqdm import tqdm
import pickle
import gc
import psutil
import sys
import argparse
from contextlib import contextmanager

# Aggressive memory optimization configuration
SENTENCE_CHUNK_SIZE = 20  # Much smaller chunks
FLOAT_DTYPE = np.float16  # Even smaller precision
MEMORY_THRESHOLD = 80  # Stop if memory usage exceeds 80%

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
        gc.collect()

def get_memory_usage():
    """Get current memory usage percentage"""
    return psutil.virtual_memory().percent

def optimize_array_dtype(arr):
    """Convert arrays to memory-efficient dtypes"""
    if arr is None:
        return arr
    if arr.dtype == np.float64:
        return arr.astype(FLOAT_DTYPE)
    elif arr.dtype == np.float32:
        return arr.astype(FLOAT_DTYPE)
    elif arr.dtype == np.int64:
        return arr.astype(np.int16)  # More aggressive int optimization
    elif arr.dtype == np.int32:
        return arr.astype(np.int16)
    return arr

def process_single_sentence(f, sentence_data, idx, subject, eeg_refs):
    """Process one sentence at a time to minimize memory usage"""
    try:
        # Get sentence string
        contentData = sentence_data['content']
        obj_reference_content = contentData[idx][0]
        sent_string = dh.load_matlab_string(f[obj_reference_content])
        
        sent_obj = {'content': sent_string}
        
        # Get sentence level EEG with memory optimization
        eeg_data = {}
        for key, refs in eeg_refs.items():
            try:
                data = np.squeeze(f[refs[idx][0]][()])
                eeg_data[key] = optimize_array_dtype(data)
                del data  # Explicit cleanup
            except (IndexError, KeyError, ValueError) as e:
                eeg_data[key] = np.array([], dtype=FLOAT_DTYPE)
        
        sent_obj['sentence_level_EEG'] = eeg_data
        sent_obj['word'] = []
        
        # Get word level data
        try:
            wordData = sentence_data['word']
            word_data, word_tokens_all, word_tokens_has_fixation, word_tokens_with_mask = \
                dh.extract_word_level_data(f, f[wordData[idx][0]])
        except (IndexError, KeyError) as e:
            print(f'Error extracting word data: subj:{subject} sentence:{idx}, skipping')
            return None
        
        if word_data == {}:
            return None
        elif len(word_tokens_all) == 0:
            return None
        else:
            for widx in range(len(word_data)):
                data_dict = word_data[widx]
                word_obj = {
                    'content': data_dict['content'], 
                    'nFixations': data_dict['nFix']
                }
                
                if 'GD_EEG' in data_dict:
                    try:
                        gd = optimize_array_dtype(np.array(data_dict["GD_EEG"], dtype=FLOAT_DTYPE))
                        ffd = optimize_array_dtype(np.array(data_dict["FFD_EEG"], dtype=FLOAT_DTYPE))
                        trt = optimize_array_dtype(np.array(data_dict["TRT_EEG"], dtype=FLOAT_DTYPE))
                        
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
                        
                        # Clean up arrays
                        del gd, ffd, trt
                        
                    except Exception as e:
                        print(f"Error processing word EEG data: {e}")
                        continue
            
            # Only keep essential data
            sent_obj['word_tokens_has_fixation'] = word_tokens_has_fixation
            sent_obj['word_tokens_with_mask'] = word_tokens_with_mask
            sent_obj['word_tokens_all'] = word_tokens_all
            
            return sent_obj
            
    except Exception as e:
        print(f"Error processing sentence {idx} for subject {subject}: {str(e)}")
        return None

def save_subject_data(subject, subject_data, output_dir, task_name):
    """Save individual subject data immediately"""
    if not subject_data:
        return
        
    subject_file = f'{task_name}-{subject}.pickle'
    subject_path = os.path.join(output_dir, subject_file)
    
    with open(subject_path, 'wb') as handle:
        pickle.dump({subject: subject_data}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved subject {subject}: {len(subject_data)} sentences -> {subject_path}')

def is_subject_already_processed(subject, output_dir, task_name):
    """Check if subject has already been processed"""
    subject_file = f'{task_name}-{subject}.pickle'
    subject_path = os.path.join(output_dir, subject_file)
    return os.path.exists(subject_path)

def get_processed_subjects(output_dir, task_name):
    """Get list of already processed subjects"""
    processed = []
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            if file.startswith(task_name) and file.endswith('.pickle') and 'partial' not in file:
                # Extract subject name from filename
                subject = file.replace(f'{task_name}-', '').replace('.pickle', '')
                if subject != 'dataset':  # Skip the final combined dataset file
                    processed.append(subject)
    return processed

def process_single_file_streaming(file_path, task, output_dir, task_name):
    """Process a single file with streaming approach - save as we go"""
    file_name = file_path
    subject = file_name.split("ts")[1].split("_")[0]
    
    # Skip YMH subject
    if subject == 'YMH':
        return subject, 0
    
    # Check if already processed
    if is_subject_already_processed(subject, output_dir, task_name):
        print(f"Subject {subject} already processed, skipping...")
        return subject, -1  # -1 indicates skipped
    
    print(f"\nProcessing subject: {subject}")
    print(f"Memory usage before: {get_memory_usage():.1f}%")
    
    sentences_processed = 0
    
    try:
        with h5py_file_manager(file_name) as f:
            sentence_data = f['sentenceData']
            rawData = sentence_data['rawData']
            total_sentences = len(rawData)
            
            print(f"  Total sentences for {subject}: {total_sentences}")
            
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
            
            subject_data = []
            
            # Process sentences in very small chunks
            for start_idx in range(0, total_sentences, SENTENCE_CHUNK_SIZE):
                end_idx = min(start_idx + SENTENCE_CHUNK_SIZE, total_sentences)
                
                # Check memory usage
                mem_usage = get_memory_usage()
                if mem_usage > MEMORY_THRESHOLD:
                    print(f"  WARNING: Memory usage high ({mem_usage:.1f}%), forcing cleanup...")
                    gc.collect()
                    mem_usage_after = get_memory_usage()
                    print(f"  Memory after cleanup: {mem_usage_after:.1f}%")
                
                print(f"  Processing sentences {start_idx}-{end_idx-1} (Memory: {get_memory_usage():.1f}%)")
                
                chunk_data = []
                for idx in range(start_idx, end_idx):
                    sent_obj = process_single_sentence(f, sentence_data, idx, subject, eeg_refs)
                    if sent_obj is not None:
                        chunk_data.append(sent_obj)
                        sentences_processed += 1
                    else:
                        chunk_data.append(None)
                    
                    # Force cleanup every few sentences
                    if idx % 5 == 0:
                        gc.collect()
                
                subject_data.extend(chunk_data)
                del chunk_data
                gc.collect()
                
                # Save partial progress every 100 sentences
                if len(subject_data) % 100 == 0 and len(subject_data) > 0:
                    print(f"  Partial save at {len(subject_data)} sentences")
                    save_subject_data(f"{subject}_partial_{len(subject_data)}", subject_data, output_dir, task_name)
            
            # Final save for this subject
            save_subject_data(subject, subject_data, output_dir, task_name)
            
    except Exception as e:
        print(f"Error processing file {file_name}: {str(e)}")
        return subject, 0
    
    print(f"  Completed subject {subject}: {sentences_processed} valid sentences")
    print(f"Memory usage after: {get_memory_usage():.1f}%")
    return subject, sentences_processed

def clean_partial_files(output_dir, task_name):
    """Remove partial files to allow reprocessing"""
    if not os.path.exists(output_dir):
        print("Output directory doesn't exist")
        return
    
    partial_files = [f for f in os.listdir(output_dir) if 'partial' in f and f.endswith('.pickle')]
    
    if not partial_files:
        print("No partial files found")
        return
    
    print(f"Found {len(partial_files)} partial files:")
    for pf in partial_files:
        print(f"  {pf}")
    
    response = input("Delete all partial files? (y/n): ")
    if response.lower() == 'y':
        for pf in partial_files:
            os.remove(os.path.join(output_dir, pf))
            print(f"Deleted {pf}")
        print("All partial files deleted")
    else:
        print("No files deleted")

def remove_subject(output_dir, task_name, subject):
    """Remove a specific subject to allow reprocessing"""
    subject_file = f'{task_name}-{subject}.pickle'
    subject_path = os.path.join(output_dir, subject_file)
    
    if os.path.exists(subject_path):
        os.remove(subject_path)
        print(f"Removed {subject} - it will be reprocessed")
    else:
        print(f"Subject {subject} not found in processed files")
    
    # Also remove any partial files for this subject
    partial_files = [f for f in os.listdir(output_dir) if f.startswith(f'{task_name}-{subject}_partial')]
    for pf in partial_files:
        os.remove(os.path.join(output_dir, pf))
        print(f"Removed partial file: {pf}")

def list_processed_subjects(output_dir, task_name):
    """List all processed subjects"""
    processed = get_processed_subjects(output_dir, task_name)
    if processed:
        print(f"Processed subjects ({len(processed)}):")
        for subject in sorted(processed):
            subject_file = f'{task_name}-{subject}.pickle'
            subject_path = os.path.join(output_dir, subject_file)
            size_mb = os.path.getsize(subject_path) / (1024*1024)
            print(f"  {subject} ({size_mb:.1f} MB)")
    else:
        print("No processed subjects found")
    """Combine individual subject files into final dataset"""
    print("\n=== Combining subject files ===")
    
    dataset_dict = {}
    subject_files = [f for f in os.listdir(output_dir) if f.startswith(task_name) and f.endswith('.pickle')]
    
    for subject_file in subject_files:
        if 'partial' in subject_file:  # Skip partial files
            continue
            
        subject_path = os.path.join(output_dir, subject_file)
        try:
            with open(subject_path, 'rb') as handle:
                subject_data = pickle.load(handle)
                dataset_dict.update(subject_data)
                print(f"Added {subject_file}")
        except Exception as e:
            print(f"Error loading {subject_file}: {e}")
    
    # Save final combined dataset
    final_path = os.path.join(output_dir, f'{task_name}-dataset.pickle')
    with open(final_path, 'wb') as handle:
        pickle.dump(dataset_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f'Final combined dataset saved: {final_path}')
    return dataset_dict

def main():
    task = "NR"
    rootdir = "./dataset/ZuCo/task2-NR-2.0/Matlab_files/"
    
    print('##############################')
    print(f'Start processing ZuCo task2-NR-2.0 (Ultra Memory Optimized)...')
    print(f'Initial memory usage: {get_memory_usage():.1f}%')
    
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
    
    # Check for already processed subjects
    processed_subjects = get_processed_subjects(output_dir, task_name)
    if processed_subjects:
        print(f"\nFound {len(processed_subjects)} already processed subjects:")
        print(f"  {processed_subjects}")
        print("These will be skipped.\n")
    
    total_subjects = 0
    total_sentences = 0
    skipped_count = 0
    
    # Process files ONE AT A TIME with immediate saving
    for i, file_path in enumerate(mat_files):
        print(f"\n=== Processing File {i+1}/{len(mat_files)} ===")
        
        subject, sentences_count = process_single_file_streaming(file_path, task, output_dir, task_name)
        
        if sentences_count == -1:  # Skipped
            skipped_count += 1
        elif sentences_count > 0:  # Successfully processed
            total_subjects += 1
            total_sentences += sentences_count
        
        # Aggressive cleanup after each file
        gc.collect()
        
        print(f"Progress: {i+1}/{len(mat_files)} files, {total_subjects} new subjects, {total_sentences} sentences, {skipped_count} skipped")
        print(f"Current memory usage: {get_memory_usage():.1f}%")
    
    # Combine all individual subject files
    final_dataset = combine_subject_files(output_dir, task_name)
    
    # Final results
    print('\n=== Final Results ===')
    print(f"Total files processed: {len(mat_files)}")
    print(f"New subjects processed: {total_subjects}")
    print(f"Subjects skipped (already done): {skipped_count}")
    print(f"Total sentences processed: {total_sentences}")
    
    if final_dataset:
        print('\nAll subjects in final dataset:', list(final_dataset.keys()))
        print('Total subjects in final dataset:', len(final_dataset))
        for subject, data in final_dataset.items():
            valid_sentences = sum(1 for item in data if item is not None)
            print(f'  {subject}: {valid_sentences} valid sentences out of {len(data)} total')
    else:
        print('No data was successfully processed!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process ZuCo EEG data with resume capability')
    parser.add_argument('--clean-partial', action='store_true', 
                       help='Clean partial files before processing')
    parser.add_argument('--remove-subject', type=str, 
                       help='Remove specific subject to reprocess (e.g., YAC)')
    parser.add_argument('--list-processed', action='store_true',
                       help='List all processed subjects')
    parser.add_argument('--combine-only', action='store_true',
                       help='Only combine existing subject files, skip processing')
    
    args = parser.parse_args()
    
    task_name = 'task2-NR-2.0'
    output_dir = f'./dataset/ZuCo/{task_name}/pickle'
    
    if args.list_processed:
        list_processed_subjects(output_dir, task_name)
        sys.exit(0)
    
    if args.clean_partial:
        clean_partial_files(output_dir, task_name)
        sys.exit(0)
    
    if args.remove_subject:
        remove_subject(output_dir, task_name, args.remove_subject)
        sys.exit(0)
    
    if args.combine_only:
        if not os.path.exists(output_dir):
            print("No output directory found")
            sys.exit(1)
        final_dataset = combine_subject_files(output_dir, task_name)
        print("Combination complete!")
        sys.exit(0)
    
    # Run main processing
    main()
