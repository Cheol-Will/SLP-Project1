import time
import numpy as np
import torch
from silero_vad import load_silero_vad


SAMPLE_RATE = 16000
VAD_INTERVAL = 1 / 20      # max 20Hz
SPEECH_THRESHOLD = 0.5     # speech probability threshold
SILENCE_DURATION = 0.6     # silence duration to trigger segment end


class VADModule:
    def __init__(self):
        self.model = load_silero_vad()
        self.model.eval()

        self._speech_buffer = np.array([], dtype=np.float32)  # accumulated audio for current segment
        self._silence_start: float = None                      # timestamp when silence started
        self._is_speech: bool = False                          # current speech state
        self._last_run: float = 0.0                            # timestamp of last VAD run
        self._segment_start: float = 0.0                       # start time of current segment

    def process(self, chunk: np.ndarray, current_time: float) -> tuple[bool, np.ndarray | None, float | None]:
        """
        chunk: newly received audio samples
        current_time: current playback time in seconds
        returns: (is_speech, completed_segment, segment_start_time)
        """
        # enforce 20Hz limit
        if current_time - self._last_run < VAD_INTERVAL:
            return self._is_speech, None, None
        self._last_run = current_time

        if chunk is None or len(chunk) == 0:
            return self._is_speech, None, None

        # Silero VAD requires 512 or 1024 samples
        window = chunk[-512:] if len(chunk) >= 512 else np.pad(chunk, (512 - len(chunk), 0))
        tensor = torch.from_numpy(window).float()

        with torch.no_grad():
            prob = self.model(tensor, SAMPLE_RATE).item()

        is_speech_now = prob > SPEECH_THRESHOLD

        if is_speech_now:
            if not self._is_speech:
                # silence -> speech transition
                self._is_speech = True
                self._silence_start = None
                self._segment_start = current_time - len(chunk) / SAMPLE_RATE
            self._speech_buffer = np.concatenate([self._speech_buffer, chunk])
        else:
            if self._is_speech:
                if self._silence_start is None:
                    self._silence_start = current_time
                elif current_time - self._silence_start >= SILENCE_DURATION:
                    # speech -> silence transition: segment complete
                    self._is_speech = False
                    completed = self._speech_buffer.copy()
                    seg_start = self._segment_start
                    self._speech_buffer = np.array([], dtype=np.float32)
                    self._silence_start = None
                    return False, completed, seg_start

        return self._is_speech, None, None

    def flush(self) -> tuple[np.ndarray | None, float | None]:
        """Force-return remaining speech buffer at end of stream."""
        if len(self._speech_buffer) > 0:
            completed = self._speech_buffer.copy()
            seg_start = self._segment_start
            self._speech_buffer = np.array([], dtype=np.float32)
            return completed, seg_start
        return None, None

    @property
    def current_buffer(self) -> np.ndarray:
        """Return current accumulated speech buffer for ASR."""
        return self._speech_buffer.copy()

    @property
    def segment_start(self) -> float:
        return self._segment_start


if __name__ == "__main__":
    import sys
    from streamer import AudioStreamer, extract_audio, CHUNK_SIZE_SEC

    video_path = sys.argv[1] if len(sys.argv) > 1 else "SLP_project01_data/clip05_raw.mp4"
    audio = extract_audio(video_path)
    streamer = AudioStreamer(audio)
    vad = VADModule()

    streamer.start()
    segments = []

    while not streamer.is_done():
        chunk = streamer.get_chunk()
        if chunk is not None and len(chunk) > 0:
            is_speech, seg, seg_start = vad.process(chunk, streamer.current_time)
            if seg is not None:
                end_time = streamer.current_time
                print(f"[VAD] Segment: {seg_start:.2f}s ~ {end_time:.2f}s  ({len(seg)/SAMPLE_RATE:.2f}s)")
                segments.append((seg_start, end_time, seg))
        time.sleep(CHUNK_SIZE_SEC)

    # flush remaining segment at end of stream
    seg, seg_start = vad.flush()
    if seg is not None:
        print(f"[VAD] Segment (flushed): {seg_start:.2f}s ~ {streamer.current_time:.2f}s")
        segments.append((seg_start, streamer.current_time, seg))

    print(f"[VAD] Total segments detected: {len(segments)}")