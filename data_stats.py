import pickle
from data import ZuCo_dataset

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

    # Load all datasets
    all_dataset_dicts = []
    for path in dataset_paths:
        with open(path, 'rb') as handle:
            all_dataset_dicts.append(pickle.load(handle))

    # You can use any tokenizer, since it's not used for counting
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2']

    train_set = ZuCo_dataset(all_dataset_dicts, 'train', tokenizer, subject='ALL', eeg_type='GD', bands=bands, setting='unique_sent')
    dev_set   = ZuCo_dataset(all_dataset_dicts, 'dev', tokenizer, subject='ALL', eeg_type='GD', bands=bands, setting='unique_sent')
    test_set  = ZuCo_dataset(all_dataset_dicts, 'test', tokenizer, subject='ALL', eeg_type='GD', bands=bands, setting='unique_sent')

    output_path = "sent_word_count.txt"
    with open(output_path, "w") as f:  # "w" will create the file if it doesn't exist, or overwrite it if it does
        for name, split in zip(['Train', 'Dev', 'Test'], [train_set, dev_set, test_set]):
            n_sent, n_words = count_sentences_and_words(split)
            f.write(f"{name}: {n_sent} sentences, {n_words} words\n")
    print(f"Output written to {output_path}")
