# SLP Project 1 — Streaming ASR Subtitle System

## Setup

```bash
uv sync
touch .env
```

Set your HuggingFace token in a `.env` file (required for `pyannote/embedding`):
```
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

Accept model access at:
- https://huggingface.co/pyannote/embedding
- https://huggingface.co/pyannote/segmentation-3.0

## Run

Place SLP_project01_data in the project root directory.

```bash
# Single clip
python main.py \
    --video SLP_project01_data/clip05_raw.mp4 \
    --spkA  SLP_project01_data/clip05_spkA_embedding.npy \
    --spkB  SLP_project01_data/clip05_spkB_embedding.npy \
    --output output_clip05.mp4 \
    --json-out pred_clip05.json

# All clips
bash main.sh
```

Output files are saved to `Result/` by default.

## Options

| Argument | Default | Description |
|---|---|---|
| `--video` | required | input MP4 path |
| `--spkA` | required | speaker A embedding `.npy` |
| `--spkB` | required | speaker B embedding `.npy` |
| `--output` | `output.mp4` | subtitle-overlaid video |
| `--json-out` | `pred.json` | committed subtitle annotation |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--compute-type` | `float16` | `float16` or `int8` |