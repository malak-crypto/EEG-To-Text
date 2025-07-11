import pickle
from data import ZuCo_dataset
from transformers import BartTokenizer
from transformers import T5Tokenizer

def count_sentences_and_words(dataset):
    num_sent = len(dataset)
    num_words = sum(dataset[i][1] for i in range(num_sent))  # dataset[i][1] is seq_len
    return num_sent, num_words

if __name__ == "__main__":
    dataset_paths = [
        './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle',
        './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle',
        './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle',
    ]
    task_names = ['Task1-SR', 'Task2-NR', 'Task2-NR-2.0']
    bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2']

    # Load pickled dicts
    all_dataset_dicts = []
    for path in dataset_paths:
        with open(path, 'rb') as handle:
            all_dataset_dicts.append(pickle.load(handle))

    tokenizer = T5Tokenizer.from_pretrained('t5-large') #to avoid downloading BART for this specific task

    output_path = "sent_word_count_per_task.txt"

    # Grand totals
    grand_total_sentences = 0
    grand_total_words = 0

    with open(output_path, "w") as f:
        for task_name, dataset_dict in zip(task_names, all_dataset_dicts):
            # Each split for this task (using single dataset_dict at a time)
            train_set = ZuCo_dataset([dataset_dict], 'train', tokenizer, subject='ALL', eeg_type='GD', bands=bands, setting='unique_sent')
            dev_set   = ZuCo_dataset([dataset_dict], 'dev', tokenizer, subject='ALL', eeg_type='GD', bands=bands, setting='unique_sent')
            test_set  = ZuCo_dataset([dataset_dict], 'test', tokenizer, subject='ALL', eeg_type='GD', bands=bands, setting='unique_sent')

            task_total_sentences = 0
            task_total_words = 0

            for split_name, split_set in zip(['Train', 'Dev', 'Test'], [train_set, dev_set, test_set]):
                n_sent, n_words = count_sentences_and_words(split_set)
                f.write(f"{task_name} {split_name}: {n_sent} sentences, {n_words} words\n")
                task_total_sentences += n_sent
                task_total_words += n_words

            f.write(f"{task_name} TOTAL: {task_total_sentences} sentences, {task_total_words} words\n\n")
            grand_total_sentences += task_total_sentences
            grand_total_words += task_total_words

        f.write(f"ALL TASKS TOTAL: {grand_total_sentences} sentences, {grand_total_words} words\n")

    print(f"Output written to {output_path}")
