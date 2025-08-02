#!/usr/bin/env python3
import os
import re
import time
import glob
import sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)

# Disable safetensors to avoid torch.frombuffer errors
os.environ["TRANSFORMERS_NO_SAFETENSORS"] = "1"

# Configure directories and pattern
DIRECTORIES = ["results_1exp"]
PATTERN = re.compile(r"^Predicted string with tf:\s*(.*)", re.IGNORECASE)

# Use a LLaMA-based model by default (override via env var)
# e.g. "NousResearch/Llama-2-7b-chat-hf" or another compatible variant
LOCAL_MODEL = os.getenv("LOCAL_RECON_MODEL", "NousResearch/Llama-2-7b-chat-hf")
DEVICE = 0 if torch.cuda.is_available() else -1

def get_system_prompt():
    # Single-line prompt to avoid string literal issues
    return (
        "[INST] You are a professional copy editor. "
        "Always rewrite the following corrupted English sentence into fluent, idiomatic English. "
        "Return only the corrected sentence—do not repeat the input or add any commentary. [/INST]"
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

# Load tokenizer and causal LLM model
print(f"Loading local model '{LOCAL_MODEL}' on {'GPU' if DEVICE == 0 else 'CPU'}...", file=sys.stderr)
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL,
    torch_dtype=dtype,
    use_safetensors=False,
)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Setup text-generation pipeline for LLaMA
reconstructor = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
    max_length=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    num_return_sequences=1,
    # use default eos_token_id
)

SYSTEM_PROMPT = get_system_prompt()

def build_prompt(text: str) -> str:
    # Wrap system prompt and few-shot examples, then the target
    prompt = SYSTEM_PROMPT + "\n\n"
    for inp, out in EXAMPLES:
        prompt += f"[INST] Corrupted: {inp}\nCorrect: {out} [/INST]\n\n"
    prompt += f"[INST] Corrupted: {text}\nCorrect: [/INST]"
    return prompt


def reconstruct_with_local(text: str) -> str:
    """
    Uses local LLaMA pipeline with chat-style, few-shot prompt.
    """
    prompt = build_prompt(text)
    try:
        outputs = reconstructor(prompt)
        gen = outputs[0].get("generated_text", "")
        # Extract everything after 'Correct:' and strip tags
        if "Correct:" in gen:
            gen = gen.split("Correct:", 1)[1]
        return gen.replace("[/INST]", "").strip()
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
    print(f"Using model {LOCAL_MODEL} on {'GPU' if DEVICE == 0 else 'CPU'}", file=sys.stderr)
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
