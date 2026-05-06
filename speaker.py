"""
speaker.py - Speaker identification using pyannote embeddings with sliding window
"""

import os
import time
import numpy as np
import torch
from pyannote.audio import Model, Inference


SAMPLE_RATE = 16000
SPEAKER_INTERVAL = 1 / 5       # max 5Hz
MIN_AUDIO_SEC = 0.5             # minimum audio length for embedding
WINDOW_SEC = 1.5                # sliding window size for speaker ID
CHANGE_THRESHOLD = 0.15         # cosine similarity drop to trigger speaker change


class SpeakerModule:
    def __init__(self, spkA_path: str, spkB_path: str):
        self.model = Model.from_pretrained(
            "pyannote/embedding",
            token=os.getenv("HF_TOKEN")
        )
        self.inference = Inference(self.model, window="whole")

        self.spkA_emb = np.load(spkA_path).flatten()  # reference embedding for speaker A
        self.spkB_emb = np.load(spkB_path).flatten()  # reference embedding for speaker B

        self._last_run: float = 0.0
        self._last_speaker: str = "A"

        # ring buffer for sliding window
        self._window_samples = int(WINDOW_SEC * SAMPLE_RATE)
        self._ring: np.ndarray = np.array([], dtype=np.float32)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        a, b = a.flatten(), b.flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _get_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from audio array."""
        tensor = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T)
        waveform = {"waveform": tensor, "sample_rate": SAMPLE_RATE}
        with torch.no_grad():
            emb = self.inference(waveform)
        return np.array(emb).flatten()

    def update(self, chunk: np.ndarray, current_time: float) -> str:
        """
        Feed new audio chunk and return current speaker.
        Uses a sliding window of WINDOW_SEC for embedding extraction.
        """
        # enforce 5Hz limit
        if current_time - self._last_run < SPEAKER_INTERVAL:
            return self._last_speaker
        self._last_run = current_time

        if chunk is None or len(chunk) == 0:
            return self._last_speaker

        # update ring buffer
        self._ring = np.concatenate([self._ring, chunk])
        if len(self._ring) > self._window_samples:
            self._ring = self._ring[-self._window_samples:]

        # skip if not enough audio
        if len(self._ring) < int(MIN_AUDIO_SEC * SAMPLE_RATE):
            return self._last_speaker

        emb = self._get_embedding(self._ring)
        sim_A = self._cosine_similarity(emb, self.spkA_emb)
        sim_B = self._cosine_similarity(emb, self.spkB_emb)

        self._last_speaker = "A" if sim_A >= sim_B else "B"
        return self._last_speaker

    def identify_segment(self, audio: np.ndarray) -> str:
        """Identify speaker for a completed VAD segment."""
        if len(audio) < int(MIN_AUDIO_SEC * SAMPLE_RATE):
            return self._last_speaker
        # use middle portion to avoid boundary noise
        mid = len(audio) // 2
        half = self._window_samples // 2
        start = max(0, mid - half)
        end = min(len(audio), mid + half)
        emb = self._get_embedding(audio[start:end])
        sim_A = self._cosine_similarity(emb, self.spkA_emb)
        sim_B = self._cosine_similarity(emb, self.spkB_emb)
        return "A" if sim_A >= sim_B else "B"

    def reset_buffer(self):
        """Clear ring buffer on speaker change or segment end."""
        self._ring = np.array([], dtype=np.float32)

    @property
    def last_speaker(self) -> str:
        return self._last_speaker


if __name__ == "__main__":
    # python speaker.py --video_path SLP_project01_data/clip05_raw.mp4 --spkA_path SLP_project01_data/clip05_spkA_embedding.npy --spkB_path SLP_project01_data/clip05_spkB_embedding.npy
    import argparse
    from streamer import AudioStreamer, extract_audio, CHUNK_SIZE_SEC
    from vad import VADModule

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--spkA_path", type=str)
    parser.add_argument("--spkB_path", type=str)
    args = parser.parse_args()

    audio = extract_audio(args.video_path)
    streamer = AudioStreamer(audio)
    vad = VADModule()
    spk = SpeakerModule(args.spkA_path, args.spkB_path)

    streamer.start()
    prev_speaker = None

    while not streamer.is_done():
        chunk = streamer.get_chunk()
        if chunk is not None and len(chunk) > 0:
            is_speech, seg, seg_start = vad.process(chunk, streamer.current_time)

            if is_speech:
                speaker = spk.update(chunk, streamer.current_time)
                if speaker != prev_speaker:
                    print(f"[t={streamer.current_time:.2f}s] Speaker change -> {speaker}")
                    prev_speaker = speaker

            if seg is not None:
                speaker = spk.identify_segment(seg)
                spk.reset_buffer()
                print(f"[VAD] Segment {seg_start:.2f}s~{streamer.current_time:.2f}s -> Speaker {speaker}")

        time.sleep(CHUNK_SIZE_SEC)

    seg, seg_start = vad.flush()
    if seg is not None:
        speaker = spk.identify_segment(seg)
        print(f"[VAD] Segment (flushed) {seg_start:.2f}s -> Speaker {speaker}")