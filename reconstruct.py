import os
import re
import time
import glob
import sys

# ——————————————————————————————————————————————————————————————
# Load your API key from the environment; ensure your SLURM script
# does: export OPENAI_API_KEY="sk-…"
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("ERROR: OPENAI_API_KEY not set in environment")
os.environ["OPENAI_API_KEY"] = api_key
# ——————————————————————————————————————————————————————————————

# Import the new, supported ChatOpenAI from langchain-community
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Prepare your LLM once for all calls
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.0,
    max_tokens=150
)

# Single system prompt
SYSTEM_PROMPT = (
    "As a text reconstructor, your task is to restore corrupted sentences to their original form "
    "while making minimum changes. Adjust spaces and punctuation as necessary. Do not introduce "
    "any additional information. If you are unable to reconstruct the text, respond with [False]."
)

# Regex to find your “Predicted string with tf:” lines
pattern = re.compile(r"^Predicted string with tf:\s*(.*)", re.IGNORECASE)

# Directories to process
directories = ["results_1exp", "results_2exp"]

def chatgpt_refinement(corrupted_text: str) -> str:
    """
    Calls the new .generate() API on ChatOpenAI to reconstruct a single sentence.
    """
    system_msg = SystemMessage(content=SYSTEM_PROMPT)
    human_msg = HumanMessage(content=f"Reconstruct the following text: [{corrupted_text}]")

    # Use the batch‐style generate interface
    response = llm.generate([[system_msg, human_msg]])
    # Extract the text of the first generation
    gen = response.generations[0][0].text.strip()

    # Strip surrounding brackets if the model emits them
    return re.sub(r"^\[|\]$", "", gen)

def main():
    for d in directories:
        if not os.path.isdir(d):
            print(f"Warning: directory '{d}' not found, skipping.")
            continue

        for infile in glob.glob(os.path.join(d, "*.txt")):
            outfile = os.path.join(d, "reconstructed_" + os.path.basename(infile))
            print(f"Processing {infile} → {outfile}")

            with open(infile, "r", encoding="utf-8") as fin, \
                 open(outfile, "w", encoding="utf-8") as fout:

                for line in fin:
                    m = pattern.match(line)
                    if m:
                        corrupted = m.group(1).strip()
                        try:
                            refined = chatgpt_refinement(corrupted)
                        except Exception as e:
                            refined = f"[ERROR] {e}"
                        fout.write(f"Original: {corrupted}\n")
                        fout.write(f"Reconstructed: {refined}\n\n")
                        # Avoid rate-limit
                        time.sleep(1)
                    else:
                        fout.write(line)
    print("Done reconstructing all sentences.")

if __name__ == "__main__":
    main()
