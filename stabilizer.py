import numpy as np

TRUNCATION_MARGIN = 3   # frames from right boundary considered unstable

class Stabilizer:
    def __init__(self, use_attention_truncation: bool = True):
        self.use_attention_truncation = use_attention_truncation
        self._prev_words: list[str] = []   # words from previous hypothesis
        self._committed: str = ""          # last committed text

    def _common_prefix_words(self, words1: list[str], words2: list[str]) -> list[str]:
        """Return longest common prefix word list between two hypotheses."""
        i = 0
        while i < len(words1) and i < len(words2) and words1[i] == words2[i]:
            i += 1
        return words1[:i]

    def attention_truncate(self, words: list[tuple], n_encoder_frames: int) -> list[tuple]:
        """
        Remove words whose peak attention is within TRUNCATION_MARGIN of right boundary.
        words: list of (word, start, end) — end mapped to encoder frame index
        n_encoder_frames: total encoder frames in current audio window
        """
        if not words or n_encoder_frames <= 0:
            return words

        stable = []
        for word, start, end in words:
            # convert timestamp to encoder frame (1 frame = 20ms for Whisper)
            peak_frame = int(end / 0.02)
            if n_encoder_frames - peak_frame >= TRUNCATION_MARGIN:
                stable.append((word, start, end))
        return stable

    def update(self, text: str, words: list[tuple], audio_len_sec: float) -> tuple[str, str]:
        """
        Update with new hypothesis and return (committed, partial).
        text: full hypothesis text
        words: list of (word, start, end)
        audio_len_sec: length of current audio buffer in seconds
        """
        current_words = text.split() if text else []

        # Step 3: attention truncation before local agreement
        if self.use_attention_truncation and words:
            n_frames = int(audio_len_sec / 0.02)  # Whisper encoder frame count
            stable_words = self.attention_truncate(words, n_frames)
            current_words = [w for w, _, _ in stable_words]

        # Step 2: local agreement
        committed_words = self._common_prefix_words(self._prev_words, current_words)
        self._prev_words = current_words

        if committed_words:
            self._committed = " ".join(committed_words)

        partial_words = current_words[len(committed_words):]
        partial = " ".join(partial_words)

        return self._committed, partial

    def commit_all(self, text: str) -> str:
        """Force commit full text at segment end."""
        self._committed = text
        self._prev_words = []
        return self._committed

    def reset(self):
        """Reset state for new segment."""
        self._prev_words = []
        self._committed = ""


if __name__ == "__main__":
    # simulate local agreement over a sequence of hypotheses
    stabilizer = Stabilizer(use_attention_truncation=False)

    hypotheses = [
        "we should",
        "we should start",
        "we should start with",
        "we should start with the",
        "we should start with the baseline",
        "we should start with the baseline first",
    ]

    print("=== Local Agreement Simulation ===")
    for i, hyp in enumerate(hypotheses):
        words = [(w, i * 0.5, i * 0.5 + 0.3) for w in hyp.split()]
        committed, partial = stabilizer.update(hyp, words, audio_len_sec=10.0)
        print(f"[step {i}] committed: '{committed}' | partial: '{partial}'")