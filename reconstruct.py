import os
import re
import time

# use LangChain for GPT-4 interface
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Pattern to extract the tf-predicted sentence
import glob
import sys
import json
from transformers import BartTokenizer
import torch

# Load your API key from environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Directories containing your result files
directories = ["results_1exp", "results_2exp"]
pattern = re.compile(r"^Predicted string with tf:\s*(.*)", re.IGNORECASE)

def chatgpt_refinement(corrupted_text: str) -> str:
    llm = ChatOpenAI(temperature=0.0, model_name="gpt-4", max_tokens=150)
    messages = [
        SystemMessage(content=(
            "As a text reconstructor, your task is to restore corrupted sentences to their original form "
            "while making minimum changes. Adjust spaces and punctuation as necessary. Do not introduce any additional information. "
            "If you are unable to reconstruct the text, respond with [False]."
        )),
        HumanMessage(content=f"Reconstruct the following text: [{corrupted_text}]")
    ]
    output = llm(messages).content.strip()
    # strip brackets if model includes them
    return output.replace("[", "").replace("]", "")

if __name__ == '__main__':
    for d in directories:
        if not os.path.isdir(d):
            print(f"Directory not found: {d}")
            continue
        for infile in glob.glob(os.path.join(d, '*.txt')):
            outfile = os.path.join(d, 'reconstructed_' + os.path.basename(infile))
            print(f"Processing {infile} -> {outfile}")
            with open(infile, 'r', encoding='utf-8') as f_in, open(outfile, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    m = pattern.match(line)
                    if m:
                        corrupted = m.group(1).strip()
                        try:
                            refined = chatgpt_refinement(corrupted)
                            f_out.write(f"Original: {corrupted}\nReconstructed: {refined}\n\n")
                            time.sleep(1)
                        except Exception as e:
                            f_out.write(f"Original: {corrupted}\nReconstructed: [ERROR] {e}\n\n")
                    else:
                        f_out.write(line)
    print("Done reconstructing all sentences.")
