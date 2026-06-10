#!/bin/bash
# Import all AttackLM models into LM Studio via lms CLI

# LM Studio scans ~/.lmstudio/models/local/ (NOT ~/.lmstudio/local/models/).
GGUF_DIR="$HOME/.lmstudio/models/local/attacklm"
PUBLISHER="attacklm"

for dir in "$GGUF_DIR"/*/; do
    model_name=$(basename "$dir")
    for gguf in "$dir"*.gguf; do
        echo "Importing $PUBLISHER/$model_name... "
        lms import "$gguf" -c --user-repo "$PUBLISHER/$model_name" -y
    done
done

echo "Done."
