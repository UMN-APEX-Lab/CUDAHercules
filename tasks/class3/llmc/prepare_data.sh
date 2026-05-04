#!/bin/bash
# Prepare TinyShakespeare data for llm.c training benchmark
# Downloads and tokenizes using GPT-2 tokenizer (tiktoken)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${1:-$SCRIPT_DIR/data}"
mkdir -p "$DATA_DIR"

TRAIN_FILE="$DATA_DIR/tiny_shakespeare_train.bin"
VAL_FILE="$DATA_DIR/tiny_shakespeare_val.bin"

if [ -f "$TRAIN_FILE" ] && [ -f "$VAL_FILE" ]; then
    echo "Data already prepared in $DATA_DIR"
    exit 0
fi

echo "Preparing TinyShakespeare data..."
python3 -c "
import tiktoken, os, urllib.request, numpy as np

DATA_DIR = '$DATA_DIR'
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
txt_file = os.path.join(DATA_DIR, 'tiny_shakespeare.txt')

if not os.path.exists(txt_file):
    print(f'Downloading {url}...')
    urllib.request.urlretrieve(url, txt_file)

enc = tiktoken.get_encoding('gpt2')
with open(txt_file, 'r') as f:
    tokens = enc.encode_ordinary(f.read())

val_tokens = tokens[:32768]
train_tokens = tokens[32768:]

def write_bin(filename, toks):
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = len(toks)
    with open(filename, 'wb') as f:
        f.write(header.tobytes())
        f.write(np.array(toks, dtype=np.uint16).tobytes())
    print(f'  {len(toks)} tokens -> {filename}')

write_bin(os.path.join(DATA_DIR, 'tiny_shakespeare_train.bin'), train_tokens)
write_bin(os.path.join(DATA_DIR, 'tiny_shakespeare_val.bin'), val_tokens)
print('Done.')
"
