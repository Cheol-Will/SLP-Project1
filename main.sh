#!/bin/bash

DATA_DIR="SLP_project01_data"
OUTPUT_DIR="result"
DEVICE="cuda"
COMPUTE_TYPE="float16"

# suppress torchcodec warnings
export PYTHONWARNINGS="ignore"

idx_list=(00 01 02 03 04 05 06 07 08 09)

for i in "${idx_list[@]}"; do
    VIDEO="$DATA_DIR/clip${i}_raw.mp4"
    SPKA="$DATA_DIR/clip${i}_spkA_embedding.npy"
    SPKB="$DATA_DIR/clip${i}_spkB_embedding.npy"
    OUTPUT="$OUTPUT_DIR/output_clip${i}.mp4"
    JSON_OUT="$OUTPUT_DIR/pred_clip${i}.json"

    echo "=== Processing clip${i} ==="

    python -W ignore main.py \
        --video "$VIDEO" \
        --spkA  "$SPKA" \
        --spkB  "$SPKB" \
        --output "$OUTPUT" \
        --json_out "$JSON_OUT" \
        --device "$DEVICE"
done

echo "=== All clips processed ==="