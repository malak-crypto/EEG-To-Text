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
LOCAL_MODEL = os.getenv("LOCAL_RECON_MODEL", "NousResearch/Llama-2-7b-chat-hf")
DEVICE = 0 if torch.cuda.is_available() else -1

# System prompt only, no few-shot examples
def get_system_prompt():
    return (
        "[INST] You are a professional copy editor. "
        "Rewrite the following corrupted English sentence into fluent, idiomatic English. "
        "Return only the corrected sentence. [/INST]"
    )

# Load tokenizer and causal LLM model
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print(f"Loading model '{LOCAL_MODEL}' on {'GPU' if DEVICE==0 else 'CPU'}...", file=sys.stderr)
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL,
    torch_dtype=dtype,
    use_safetensors=False,
)
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Setup text-generation pipeline
reconstructor = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=DEVICE,
    max_length=256,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    num_return_sequences=1,
)

SYSTEM_PROMPT = get_system_prompt()

def reconstruct_with_local(text: str) -> str:
    """
    Uses local LLaMA pipeline with system prompt only.
    """
    prompt = f"{SYSTEM_PROMPT}\n[INST] Corrupted: {text} [/INST]"
    try:
        output = reconstructor(prompt)[0]["generated_text"]
        # Strip prompt prefix
        corrected = output.replace(prompt, "").strip()
        # In case tags remain
        corrected = corrected.replace("[/INST]", "").strip()
        return corrected
    except Exception as e:
        return f"[ERROR] {e}"


def process_file(infile: str, outfile: str):
    """
    Read infile line by line, reconstruct matches and write only original + reconstructed.
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
                # copy other lines unchanged
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
    print("Done.")

if __name__ == '__main__':
    main()
