import os
import re
import time
import openai

# Instructions to obtain your OpenAI API key:
# 1. Go to https://platform.openai.com/account/api-keys
# 2. Click on "Create new secret key" and copy the generated key.
# 3. Set it as an environment variable:
export OPENAI_API_KEY="your_api_key_here"

# Load your API key from environment
openai.api_key = os.getenv("OPEN_API_KEY")
if openai.api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Directories containing your result files
directories = ["results_1exp", "results_2exp"]
# Pattern to extract the tf-predicted sentence
pattern = re.compile(r"^predicted string with tf:\s*(.*)", re.IGNORECASE)

# Function to call the OpenAI API and reconstruct the sentence
 def reconstruct_sentence(sentence: str) -> str:
    """
    Sends the given corrupted sentence to the OpenAI Chat API and attempts to restore it
    to its original form with minimal changes.
    Returns the reconstructed sentence or "[False]" if it cannot be reconstructed.
    """
    system_prompt = (
        "As a text reconstructor, your task is to restore corrupted sentences to their original form while making minimum changes. "
        "You should adjust the spaces and punctuation marks as necessary. Do not introduce any additional information. "
        "If you are unable to reconstruct the text, respond with [False]."
    )
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sentence}
        ],
        temperature=0.0,
        max_tokens=150,
    )
    # Extract the assistant's reply
    new_sentence = response.choices[0].message.content.strip()
    return new_sentence

if __name__ == '__main__':
    # Loop through each directory and file
    for d in directories:
        if not os.path.isdir(d):
            print(f"Directory not found: {d}")
            continue

        for filename in os.listdir(d):
            if filename.endswith('.txt'):
                infile = os.path.join(d, filename)
                outfile = os.path.join(d, f"reconstructed_{filename}")
                print(f"Processing {infile} -> {outfile}")

                with open(infile, 'r', encoding='utf-8') as f_in, \
                     open(outfile, 'w', encoding='utf-8') as f_out:
                    for line in f_in:
                        match = pattern.match(line)
                        if match:
                            original = match.group(1).strip()
                            try:
                                reconstructed = reconstruct_sentence(original)
                                f_out.write(f"Original: {original}\n")
                                f_out.write(f"Reconstructed: {reconstructed}\n\n")
                                # be kind to the API
                                time.sleep(1)
                            except Exception as e:
                                print(f"Error reconstructing sentence '{original}': {e}")
                                f_out.write(f"Original: {original}\n")
                                f_out.write("Reconstructed: [ERROR]" + str(e) + "\n\n")
                        else:
                            # copy other lines as-is
                            f_out.write(line)

    print("Done reconstructing all sentences.")
