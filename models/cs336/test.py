import os
import requests
from tokenizer import BPETokenizerSimple


def download_file_if_absent(url, filename, search_dirs):
    for directory in search_dirs:
        file_path = os.path.join(directory, filename)
        if os.path.exists(file_path):
            print(f"{filename} already exists in {file_path}")
            return file_path

    target_path = os.path.join(search_dirs[0], filename)
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(target_path, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    out_file.write(chunk)
        print(f"Downloaded {filename} to {target_path}")
    except Exception as e:
        print(f"Failed to download {filename}. Error: {e}")

    return target_path


verdict_path = download_file_if_absent(
    url=(
        "https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    ),
    filename="the-verdict.txt",
    search_dirs=["."],
)

with open(verdict_path, "r", encoding="utf-8") as f:  # added ../01_main-chapter-code/
    text = f.read()

tokenizer = BPETokenizerSimple()
tokenizer.train(text, vocab_size=1000, allowed_special={"<|endoftext|>"})

# print(tokenizer.vocab)
print(len(tokenizer.vocab))

print(len(tokenizer.bpe_merges))

input_text = "Jack embraced beauty through art and life."
token_ids = tokenizer.encode(input_text)
print(token_ids)

input_text = "Jack embraced beauty through art and life.<|endoftext|> "
token_ids = tokenizer.encode(input_text, allowed_special={"<|endoftext|>"})
print(token_ids)

print("Number of characters:", len(input_text))
print("Number of token IDs:", len(token_ids))
