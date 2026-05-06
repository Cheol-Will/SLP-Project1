# Spoken Language Processing Assignment #2

This project aims to design a streaming ASR subtitle system using existing modules.

- Input: a conversation between two people
- Each person’s speaker embedding will be provided in advance.
- Output 1: Live subtitle display: committed subtitle + partial subtitle
- Output 2: Final subtitle file / JSON annotation: committed subtitles

Dataset:
- Benchmark: Seamless Interaction
- over 4,000 hours of face-to-face interaction footage designed for modeling human interaction and communication.


The final system should combine VAD, speaker embedding, Whisper ASR, and subtitle stabilization: 
1. Extract audio chunks in real time
2. VAD: detect speech regions
3. Speaker Embedding: assign Speaker A or Speaker B
4. Whisper ASR: decode buffered speech
5. Subtitle stabilization
6. Display partial and committed subtitles

Simulation constraints:
- Consider a practical on-device ASR system.
- Assume the maximum ASR frequency is 2 Hz for small.en and 1 Hz for medium.en.
- Assume the maximum VAD execution frequency is 20 Hz.
- Assume the maximum Speaker embedding model execution frequency is 5 Hz.
- The system must not access future audio beyond the current video playback time.

### Dependency
```bash
uv venv
source .venv/bin/activate
uv pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
uv pip install transformers jiwer wandb python-dotenv tqdm matplotlib
```

### Wandb setup

To track training and validation statistics, word error rate (with greedy decoding), learning rate, and gradient norm, add your wandb api key in a `.env` file in the root directory as follows:
```
WANDB_API_KEY=YOUR_API_KEY
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

### Run the experiment
```bash
bash main.sh
```