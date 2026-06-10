#!/bin/bash
# Recursively import all .gguf files into LM Studio
# Usage: ./import_gguf.sh /path/to/gguf/files

DIR="${1:-.}"

find "$DIR" -name "*.gguf" | while read -r gguf; do
    modelname=$(basename "$(dirname "$gguf")")
    echo "importing $gguf"
    lms import "$gguf" -c -y --user-repo "local/$modelname"
done
