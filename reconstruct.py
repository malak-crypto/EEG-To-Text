import os
import re
import time
import glob
import sys
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Configure directories and pattern
DIRECTORIES = ["results_1exp", "results_2exp"]
PATTERN = re.compile(r"^Predicted string with tf:\s*(.*)", re.IGNORECASE)

# Choose a free local model via env var, default to 't5-small'
LOCAL_MODEL = os.getenv("LOCAL_RECON_MODEL", "t5-small")
# Determine device: GPU if available
DEVICE = 0 if torch.cuda.is_available() else -1

# Load tokenizer and model manually to disable safetensors
print(f"Loading local model '{LOCAL_MODEL}' on {'GPU' if DEVICE==0 else 'CPU'}...", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL, use_safetensors=False)

# Setup the text2text pipeline with beam search parameters
reconstructor = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
    max_length=128,
    do_sample=False,
    num_beams=5,
    early_stopping=True,
)

# Enhanced prompt
SYSTEM_PROMPT = (
    "You are a linguistic expert specialized in restoring corrupted or garbled sentences. "
    "Given a distorted English sentence, correct all grammatical errors, fix word order, punctuation, and spacing, "
    "while preserving the original meaning and without adding new information. "
    "Respond only with the corrected sentence."
)

def reconstruct_with_local(text: str) -> str:
    """
    Uses the local HF pipeline to reconstruct corrupted text with beam search.
    """
    prompt = f"""{SYSTEM_PROMPT}
Corrupted sentence: \"{text}\"
Corrected sentence:"""
    try:
        result = reconstructor(prompt)[0]
        return result["generated_text"].strip()
    except Exception as e:
        return f"[ERROR] {e}"


def process_file(infile: str, outfile: str):
    """
    Read infile line by line, reconstruct matches and write to outfile.
    """
    with open(infile, 'r', encoding='utf-8') as fin, \
         open(outfile, 'w', encoding='utf-8') as fout:
        for line in fin:
            m = PATTERN.match(line)
            if m:
                corrupted = m.group(1).strip()
                reconstructed = reconstruct_with_local(corrupted)
                fout.write(f"Original: {corrupted}\nReconstructed: {reconstructed}\n\n")
                time.sleep(0.5)
            else:
                fout.write(line)


def main():
    print(f"Processing directories: {DIRECTORIES}", file=sys.stderr)
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
