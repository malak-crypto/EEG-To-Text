import os
import re
import time
import glob
import sys
import torch
from transformers import pipeline

# Configure directories and pattern
DIRECTORIES = ["results_1exp", "results_2exp"]
PATTERN = re.compile(r"^predicted string with tf:\s*(.*)", re.IGNORECASE)

# Choose a free local model
# Options: 'google/flan-t5-base', 'google/flan-t5-large'
LOCAL_MODEL = os.getenv("LOCAL_RECON_MODEL", "google/t5-small")

# Initialize the text2text pipeline once
device = 0 if torch.cuda.is_available() else -1
reconstructor = pipeline(
    "text2text-generation",
    model=LOCAL_MODEL,
    device=device,
    max_length=128,
    do_sample=False,
)

# System prompt for local model
SYSTEM_PROMPT = (
    "Restore this corrupted sentence to its original form with minimal edits. "
    "Adjust spaces and punctuation as necessary without adding new information."
)


def reconstruct_with_local(text: str) -> str:
    """
    Uses the Hugging Face pipeline to reconstruct corrupted text.
    """
    prompt = f"{SYSTEM_PROMPT} Text: \"{text}\""
    try:
        result = reconstructor(prompt)[0]
        return result["generated_text"].strip()
    except Exception as e:
        return f"[ERROR] {e}"


def process_file(infile: str, outfile: str):
    """
    Read infile line by line, reconstructing matches and writing to outfile.
    """
    with open(infile, 'r', encoding='utf-8') as fin, \
         open(outfile, 'w', encoding='utf-8') as fout:
        for line in fin:
            m = PATTERN.match(line)
            if m:
                corrupted = m.group(1).strip()
                reconstructed = reconstruct_with_local(corrupted)
                fout.write(f"Original: {corrupted}\nReconstructed: {reconstructed}\n\n")
                time.sleep(0.5)  # rate-limit locally
            else:
                fout.write(line)


def main():
    # Verify model
    print(f"Using local model: {LOCAL_MODEL} (device: {'cuda' if device==0 else 'cpu'})", file=sys.stderr)
    for d in DIRECTORIES:
        if not os.path.isdir(d):
            print(f"Warning: directory '{d}' not found, skipping.")
            continue
        for infile in glob.glob(os.path.join(d, '*.txt')):
            outfile = os.path.join(d, 'reconstructed_' + os.path.basename(infile))
            print(f"Processing {infile} -> {outfile}")
            process_file(infile, outfile)
    print("Done reconstructing all sentences.")


if __name__ == '__main__':
    main()
