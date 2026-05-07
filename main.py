import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import os
import time
import json
import argparse
import numpy as np
import cv2

from streamer import AudioStreamer, extract_audio, CHUNK_SIZE_SEC, SAMPLE_RATE
from vad import VADModule
from speaker import SpeakerModule
from asr import ASRModule
from stabilizer import Stabilizer


def render_subtitle_frame(frame: np.ndarray, speaker: str, committed: str, partial: str) -> np.ndarray:
    """Render committed and partial subtitles onto a video frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    committed_text = f"{speaker}: {committed}" if committed else ""
    partial_text = f"({partial})" if partial else ""

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(overlay, committed_text, (20, h - 60), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(overlay, partial_text,   (20, h - 25), font, 0.6, (180, 180, 180), 1, cv2.LINE_AA)

    return overlay


def get_video_properties(video_path: str) -> tuple[int, int, float]:
    """Return (width, height, fps) of video."""
    cap = cv2.VideoCapture(video_path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return w, h, fps


def main(args):
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    json_dir = os.path.dirname(args.json_out)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)

    audio         = extract_audio(args.video)
    w, h, fps     = get_video_properties(args.video)

    streamer   = AudioStreamer(audio)
    vad        = VADModule()
    spk        = SpeakerModule(args.spkA, args.spkB)
    asr        = ASRModule(device=args.device, compute_type=args.compute_type)
    stabilizer = Stabilizer(use_attention_truncation=True)

    cap    = cv2.VideoCapture(args.video)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    committed_subtitles = []
    current_speaker     = "A"
    current_seg_start   = 0.0
    prev_speaker        = None
    seg_committed_text  = ""
    last_written_frame  = -1   # track last written frame to avoid gaps/duplicates

    streamer.start()

    while not streamer.is_done():
        chunk = streamer.get_chunk()
        ct    = streamer.current_time

        committed = seg_committed_text
        partial   = ""

        if chunk is not None and len(chunk) > 0:
            is_speech, seg, seg_start = vad.process(chunk, ct)

            if is_speech:
                current_speaker = spk.update(chunk, ct)

                if prev_speaker is not None and current_speaker != prev_speaker:
                    if seg_committed_text and seg_committed_text.strip():
                        committed_subtitles.append({
                            "speaker":     prev_speaker,
                            "start":       round(current_seg_start, 2),
                            "end":         round(ct, 2),
                            "commit_time": round(ct + 1.2, 2),
                            "text":        seg_committed_text.strip(),
                        })
                    stabilizer.reset()
                    spk.reset_buffer()
                    current_seg_start  = ct
                    seg_committed_text = ""

                prev_speaker = current_speaker

                text, words = asr.transcribe(vad.current_buffer, ct)
                committed, partial = stabilizer.update(text, words, len(vad.current_buffer) / SAMPLE_RATE)
                seg_committed_text = committed

            if seg is not None:
                final_text, _ = asr.transcribe_segment(seg)
                final_text     = stabilizer.commit_all(final_text)
                seg_committed_text = final_text
                committed          = final_text

                if final_text.strip():
                    committed_subtitles.append({
                        "speaker":     current_speaker,
                        "start":       round(seg_start, 2),
                        "end":         round(ct, 2),
                        "commit_time": round(ct + 1.2, 2),
                        "text":        final_text.strip(),
                    })
                stabilizer.reset()
                spk.reset_buffer()
                seg_committed_text = ""
                prev_speaker       = None

                print(f"[COMMIT] {seg_start:.2f}s~{ct:.2f}s [{current_speaker}]: {final_text}")

        # write every frame from last written up to current time
        target_frame = int(ct * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, last_written_frame + 1)
        for fi in range(last_written_frame + 1, target_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            frame = render_subtitle_frame(frame, current_speaker, committed, partial)
            out.write(frame)
            last_written_frame = fi

        time.sleep(CHUNK_SIZE_SEC)

    # flush remaining segment
    seg, seg_start = vad.flush()
    if seg is not None:
        final_text, _ = asr.transcribe_segment(seg)
        ct = streamer.current_time
        if final_text.strip():
            committed_subtitles.append({
                "speaker":     current_speaker,
                "start":       round(seg_start, 2),
                "end":         round(ct, 2),
                "commit_time": round(ct + 1.2, 2),
                "text":        final_text.strip(),
            })
        print(f"[COMMIT] (flushed) {seg_start:.2f}s [{current_speaker}]: {final_text}")

    # write any remaining frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = render_subtitle_frame(frame, current_speaker, seg_committed_text, "")
        out.write(frame)

    cap.release()
    out.release()

    with open(args.json_out, "w") as f:
        json.dump(committed_subtitles, f, indent=2)

    print(f"[Done] Video saved to {args.output}")
    print(f"[Done] JSON saved to {args.json_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",        type=str, required=True)
    parser.add_argument("--spkA",         type=str, required=True)
    parser.add_argument("--spkB",         type=str, required=True)
    parser.add_argument("--output",       type=str, default="output.mp4")
    parser.add_argument("--json_out",     type=str, default="pred.json")
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--compute-type", type=str, default="float16")
    args = parser.parse_args()
    main(args)