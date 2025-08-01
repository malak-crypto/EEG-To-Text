import os
import re
import time
import glob
import sys
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Configure directories and pattern
DIRECTORIES = ["results_1exp"]
PATTERN = re.compile(r"^Predicted string with tf:\s*(.*)", re.IGNORECASE)

# Use Flan-T5-large by default (override via env var)
LOCAL_MODEL = os.getenv("LOCAL_RECON_MODEL", "google/flan-t5-large")
DEVICE = 0 if torch.cuda.is_available() else -1

# Load tokenizer and model manually to disable safetensors
print(f"Loading local model '{LOCAL_MODEL}' on {'GPU' if DEVICE==0 else 'CPU'}...", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, use_fast=True)
model = AutoModelForSeq2SeqLM.from_pretrained(LOCAL_MODEL, use_safetensors=False)

# Setup the text2text pipeline using beam search
reconstructor = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
    max_length=256,
    do_sample=False,
    num_beams=5,
    early_stopping=True,
)

# Few-shot examples to guide the model
EXAMPLES = [
    (
        "He of the film series, recognize to but the will will be happy disappointed.",
        "Fans of the film series will recognize him, but they will ultimately be disappointed."
    ),
    (
        "to the, who of movies things in possible into a minutes. a most of a exerciseagragger, of time time",
        "For those who love movies, it’s amazing how much you can pack into just a few minutes—a real exercise in storytelling brevity."
    ),
    (
        "The toa to the for who the best characters in in the minutes. and first of lyposed of time.",
        "The best characters appear within minutes, and that’s only the beginning of this fast-paced adventure."
    ),
    (
        "The of the film series, remember pleased by but the will be disappointed disappointed.",
        "Fans of the film series will be pleased by the nostalgia, yet they may still end up disappointed."
    ),
]

SYSTEM_PROMPT = (
    "You are an expert in correcting corrupted English sentences. "
    "Given several examples and a new corrupted sentence, produce only the corrected sentence."
)

def reconstruct_with_local(text: str) -> str:
    """
    Uses local HF pipeline with few-shot examples.
    """
    prompt = SYSTEM_PROMPT + "\n\n"
    for inp, out in EXAMPLES:
        prompt += f"Corrupted: {inp}\nCorrect: {out}\n\n"
    prompt += f"Corrupted: {text}\nCorrect:"
    try:
        res = reconstructor(prompt)[0]
        return res["generated_text"].strip()
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
                corrected = reconstruct_with_local(corrupted)
                fout.write(f"Original: {corrupted}\nReconstructed: {corrected}\n\n")
                time.sleep(0.5)
            else:
                fout.write(line)

def main():
    print(f"Using model {LOCAL_MODEL} on {'GPU' if DEVICE==0 else 'CPU'}", file=sys.stderr)
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
