import time
import numpy as np
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
ASR_INTERVAL = 1 / 2
MIN_AUDIO_SEC = 1.0 


class ASRModule:
    def __init__(self, model_size: str = "small.en", device: str = "cuda", compute_type: str = "float16"):
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._last_run: float = 0.0
        self._last_text: str = ""

    def transcribe(self, audio: np.ndarray, current_time: float) -> tuple[str, list]:
        """
        Transcribe audio buffer and return (text, word_timestamps).
        word_timestamps: list of (word, start, end, tokens, avg_logprob)
        """
        # enforce 2Hz limit
        if current_time - self._last_run < ASR_INTERVAL:
            return self._last_text, []
        self._last_run = current_time

        if audio is None or len(audio) < int(MIN_AUDIO_SEC * SAMPLE_RATE):
            return self._last_text, []

        segments, _ = self.model.transcribe(
            audio,
            language="en",
            word_timestamps=True,
            vad_filter=False,
        )

        words = []
        text_parts = []
        for seg in segments:
            text_parts.append(seg.text.strip())
            if seg.words:
                for w in seg.words:
                    words.append((w.word.strip(), w.start, w.end))

        self._last_text = " ".join(text_parts).strip()
        return self._last_text, words

    def transcribe_segment(self, audio: np.ndarray) -> tuple[str, list]:
        """Transcribe a completed VAD segment without rate limiting."""
        if audio is None or len(audio) < int(MIN_AUDIO_SEC * SAMPLE_RATE):
            return "", []

        segments, _ = self.model.transcribe(
            audio,
            language="en",
            word_timestamps=True,
            vad_filter=False,
        )

        words = []
        text_parts = []
        for seg in segments:
            text_parts.append(seg.text.strip())
            if seg.words:
                for w in seg.words:
                    words.append((w.word.strip(), w.start, w.end))

        text = " ".join(text_parts).strip()
        self._last_text = text
        return text, words


if __name__ == "__main__":
    # python asr.py --video SLP_project01_data/clip05_raw.mp4
    import argparse
    from streamer import AudioStreamer, extract_audio, CHUNK_SIZE_SEC
    from vad import VADModule

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="SLP_project01_data/clip05_raw.mp4")
    args = parser.parse_args()
    video_path = args.video

    audio = extract_audio(video_path)
    streamer = AudioStreamer(audio)
    vad = VADModule()
    asr = ASRModule()

    streamer.start()

    while not streamer.is_done():
        chunk = streamer.get_chunk()
        if chunk is not None and len(chunk) > 0:
            is_speech, seg, seg_start = vad.process(chunk, streamer.current_time)

            if is_speech:
                text, words = asr.transcribe(vad.current_buffer, streamer.current_time)
                if text:
                    print(f"[t={streamer.current_time:.2f}s] partial: {text}")

            if seg is not None:
                text, words = asr.transcribe_segment(seg)
                print(f"[ASR] {seg_start:.2f}s~{streamer.current_time:.2f}s: {text}")

        time.sleep(CHUNK_SIZE_SEC)

    seg, seg_start = vad.flush()
    if seg is not None:
        text, words = asr.transcribe_segment(seg)
        print(f"[ASR] (flushed) {seg_start:.2f}s: {text}")