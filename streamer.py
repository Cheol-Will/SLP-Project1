import time
import numpy as np
import av


SAMPLE_RATE = 16000       # Hz — Whisper expects 16kHz mono
CHUNK_SIZE_SEC = 0.5      # seconds per chunk
MAX_BUFFER_SEC = 30.0     # max ASR audio buffer length (Whisper limit)


def extract_audio(video_path: str) -> np.ndarray:
    """
    Extract mono 16kHz audio from a video file using PyAV (no ffmpeg binary needed).
    Returns a float32 numpy array normalized to [-1, 1].
    """
    container = av.open(video_path)
    audio_stream = next(s for s in container.streams if s.type == "audio")

    resampler = av.AudioResampler(
        format="fltp",
        layout="mono",
        rate=SAMPLE_RATE,
    )

    chunks = []
    for frame in container.decode(audio_stream):
        resampled = resampler.resample(frame)
        for r in resampled:
            arr = r.to_ndarray()  # shape: (1, N)
            chunks.append(arr[0])

    container.close()
    audio = np.concatenate(chunks).astype(np.float32)
    return audio


class AudioStreamer:
    """Simulates real-time audio streaming from a pre-loaded audio array."""

    def __init__(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                 chunk_size_sec: float = CHUNK_SIZE_SEC):
        self.audio = audio
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_size_sec)
        self.total_samples = len(audio)
        self.total_duration = self.total_samples / sample_rate

        self._start_time: float = None
        self._last_delivered: int = 0  # sample index of last delivered position

    def start(self):
        """Start the streaming clock."""
        self._start_time = time.time()
        self._last_delivered = 0
        print(f"[Streamer] Started. Total duration: {self.total_duration:.2f}s")

    @property
    def current_time(self) -> float:
        """Wall-clock elapsed time since start (= current playback time)."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def available_samples(self) -> int:
        """Number of audio samples available up to current playback time."""
        return min(int(self.current_time * self.sample_rate), self.total_samples)

    def get_chunk(self) -> np.ndarray | None:
        """
        Returns the next chunk of audio available since last call.
        Returns None if no new samples are available yet.
        Enforces: never returns audio beyond current playback time.
        """
        avail = self.available_samples
        if avail <= self._last_delivered:
            return None  # no new audio yet

        chunk = self.audio[self._last_delivered:avail]
        self._last_delivered = avail
        return chunk

    def get_buffer(self, max_sec: float = MAX_BUFFER_SEC) -> np.ndarray:
        """Returns up to max_sec seconds of audio ending at current playback time."""
        avail = self.available_samples
        max_samples = int(max_sec * self.sample_rate)
        start = max(0, avail - max_samples)
        return self.audio[start:avail]

    def is_done(self) -> bool:
        """Returns True when all audio has been delivered."""
        return self._last_delivered >= self.total_samples


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python streamer.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    print(f"[Streamer] Extracting audio from: {video_path}")
    audio = extract_audio(video_path)
    print(f"[Streamer] Audio loaded: {len(audio)} samples, {len(audio)/SAMPLE_RATE:.2f}s")

    streamer = AudioStreamer(audio)
    streamer.start()

    total_chunks = 0
    while not streamer.is_done():
        chunk = streamer.get_chunk()
        if chunk is not None and len(chunk) > 0:
            total_chunks += 1
            print(f"[t={streamer.current_time:.2f}s] chunk: {len(chunk)} samples "
                  f"({len(chunk)/SAMPLE_RATE:.2f}s)")
        time.sleep(CHUNK_SIZE_SEC)

    print(f"[Streamer] Done. Total chunks delivered: {total_chunks}")