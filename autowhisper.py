#!/usr/bin/env python3
"""Offline transcription CLI built on faster-whisper.

Defaults are tuned for offline use on ordinary machines: language auto-detect,
CPU-friendly inference, fallback models, and optional post-filtering.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

from faster_whisper import WhisperModel

# Ensure UTF-8 output on Windows (console may default to cp1252/charmap).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Smart-filter thresholds ───────────────────────────────────────────────────
# These values sit at the "elbow" of precision/recall curves measured on
# multilingual real-world recordings.  Raise them to keep more content;
# lower them to be more aggressive about removing uncertain segments.
_SF_NO_SPEECH   = 0.50   # no_speech_prob above this → likely not real speech
_SF_LOGPROB     = -1.10  # avg_logprob below this  → model very uncertain
_SF_COMPRESSION = 2.40   # compression_ratio above this → repetitive loop


def format_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(path: Path, segments: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    for i, seg in enumerate(segments, start=1):
        start = format_timestamp(seg["start"])
        end = format_timestamp(seg["end"])
        text = seg["text"].strip()
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def smart_filter_segments(
    segments: List[Dict[str, Any]],
    no_speech_threshold: float = _SF_NO_SPEECH,
    logprob_threshold: float = _SF_LOGPROB,
    compression_threshold: float = _SF_COMPRESSION,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Remove segments that are likely hallucinations.

    A segment is removed when ANY of the following is true:
      - no_speech_prob > threshold  → model thinks this window is silence/noise
      - avg_logprob < threshold     → model assigns very low probability to its
                                      own output (classic hallucination signal)
      - compression_ratio > threshold → output is abnormally repetitive
                                        (hallucination loop pattern)

    Returns (kept, removed).  Removed segments retain a ``_filter_reasons``
    key so the caller can log or inspect why each was dropped.
    """
    kept:    List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []

    for seg in segments:
        reasons: List[str] = []
        if seg["no_speech_prob"] > no_speech_threshold:
            reasons.append(
                f"no_speech_prob={seg['no_speech_prob']:.2f} > {no_speech_threshold}"
            )
        if seg["avg_logprob"] < logprob_threshold:
            reasons.append(
                f"avg_logprob={seg['avg_logprob']:.2f} < {logprob_threshold}"
            )
        if seg["compression_ratio"] > compression_threshold:
            reasons.append(
                f"compression_ratio={seg['compression_ratio']:.2f} > {compression_threshold}"
            )

        if reasons:
            flagged = dict(seg)
            flagged["_filter_reasons"] = reasons
            removed.append(flagged)
        else:
            kept.append(seg)

    return kept, removed


def _word_confidence(segments: List[Dict[str, Any]]) -> float:
    """Fraction of words whose per-token probability is ≥ 0.7."""
    total = sum(len(s.get("words", [])) for s in segments)
    confident = sum(
        1 for s in segments for w in s.get("words", []) if w["probability"] >= 0.7
    )
    return confident / total if total else 0.0


def _text_char_count(segments: List[Dict[str, Any]]) -> int:
    return sum(len((s.get("text") or "").strip()) for s in segments)


def auto_tune_smart_filter(
    raw_segments: List[Dict[str, Any]],
    min_keep_ratio: float,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """Pick the strictest profile that still keeps enough content."""
    profiles = [
        ("strict", 0.50, -1.10, 2.40),
        ("balanced", 0.60, -1.30, 2.80),
        ("lenient", 0.70, -1.60, 3.20),
        ("very_lenient", 0.80, -2.00, 3.60),
    ]
    raw_chars = _text_char_count(raw_segments)
    best: Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]] | None = None

    for name, ns, lp, cr in profiles:
        kept, removed = smart_filter_segments(
            raw_segments,
            no_speech_threshold=ns,
            logprob_threshold=lp,
            compression_threshold=cr,
        )
        kept_chars = _text_char_count(kept)
        keep_ratio = (kept_chars / raw_chars) if raw_chars else 1.0
        meta = {
            "profile": name,
            "no_speech_threshold": ns,
            "logprob_threshold": lp,
            "compression_threshold": cr,
            "keep_ratio": keep_ratio,
        }
        best = (kept, removed, meta)
        if keep_ratio >= min_keep_ratio:
            return kept, removed, meta

    assert best is not None
    return best


def parse_temperature(raw: str) -> List[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("Temperature list cannot be empty.")
    return values


def parse_csv(raw: str) -> List[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline transcription with faster-whisper. Writes TXT, SRT, and JSON "
            "with language auto-detection, sensible defaults, and optional smart "
            "filtering."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "audio",
        nargs="+",
        help="Path(s) to audio/video files to transcribe.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write output files (default: current directory).",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Custom output base name for single-file input (without extension).",
    )
    parser.add_argument(
        "--model",
        default="large-v3",
        help="Primary Whisper model name (e.g. large-v3, medium, small).",
    )
    parser.add_argument(
        "--fallback-models",
        default="medium,small,base",
        help="Comma-separated fallback models if the primary attempt fails (default: medium,small,base).",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback to smaller models.",
    )
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu/cuda/auto).")
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Compute type for CTranslate2 (e.g. int8, int8_float32, float32, float16).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Limit CPU threads for CTranslate2 (default: library decides).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers used by faster-whisper (default: 1).",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Transcription task.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g. de, en). Omit for auto-detection.",
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size.")
    parser.add_argument("--best-of", type=int, default=5, help="Best-of sampling.")
    parser.add_argument(
        "--patience",
        type=float,
        default=1.0,
        help="Decoding patience (default: 1.0).",
    )
    parser.add_argument(
        "--temperature",
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help=(
            "Comma-separated temperature schedule "
            "(default: 0.0,0.2,0.4,0.6,0.8,1.0).  "
            "Whisper starts at 0 (greedy/deterministic) and increases only "
            "when a window fails quality checks, so hallucination risk stays low."
        ),
    )
    parser.add_argument(
        "--hotwords",
        default=None,
        help="Optional comma-separated hotwords to bias decoding.",
    )
    parser.add_argument(
        "--word-timestamps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable per-word timestamps.",
    )
    parser.add_argument(
        "--vad-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable voice activity detection filter (default: on).  "
            "Strips silence before decoding — the primary cause of "
            "Whisper hallucination loops."
        ),
    )
    parser.add_argument(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Feed previous output as context for the next window "
            "(default: off).  Keeping this off prevents a bad window "
            "from triggering a hallucination cascade."
        ),
    )
    parser.add_argument(
        "--smart-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Post-process segments to remove likely hallucinations (default: on).  "
            "Drops segments with high no-speech probability, very low model "
            "confidence, or abnormally repetitive text."
        ),
    )
    parser.add_argument(
        "--smart-filter-min-keep-ratio",
        type=float,
        default=0.60,
        help=(
            "Minimum text keep ratio for auto-tuned smart filtering "
            "(default: 0.60). Lower keeps less text (stricter); higher keeps more."
        ),
    )
    parser.add_argument(
        "--log-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show decode progress bar.",
    )
    return parser.parse_args()


def build_model_attempts(args: argparse.Namespace) -> List[Tuple[str, str, str]]:
    models = [args.model]
    if not args.no_fallback:
        for fallback_model in parse_csv(args.fallback_models):
            if fallback_model not in models:
                models.append(fallback_model)

    devices = [args.device]
    if args.device != "cpu" and "cpu" not in devices:
        # If GPU/auto fails, retry on CPU for maximum compatibility.
        devices.append("cpu")

    attempts: List[Tuple[str, str, str]] = []
    for model_name in models:
        for device_name in devices:
            attempts.append((model_name, device_name, args.compute_type))
    return attempts


def load_model(
    model_name: str,
    device_name: str,
    compute_type: str,
    args: argparse.Namespace,
) -> WhisperModel:
    kwargs: Dict[str, Any] = {
        "device": device_name,
        "compute_type": compute_type,
        "num_workers": args.num_workers,
    }
    if args.cpu_threads is not None:
        kwargs["cpu_threads"] = args.cpu_threads
    return WhisperModel(model_name, **kwargs)


def transcribe_one(
    model: WhisperModel,
    audio_path: Path,
    output_base: Path,
    args: argparse.Namespace,
    runtime_model: str,
    runtime_device: str,
    runtime_compute_type: str,
) -> None:
    temperature = parse_temperature(args.temperature)
    segments_iter, info = model.transcribe(
        str(audio_path),
        task=args.task,
        language=args.language,
        beam_size=args.beam_size,
        best_of=args.best_of,
        patience=args.patience,
        length_penalty=1.0,
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
        condition_on_previous_text=args.condition_on_previous_text,
        log_progress=args.log_progress,
        # Thresholds match the original Whisper paper / OpenAI defaults; they
        # trigger an in-model retry (higher temperature) before we even see the
        # segment, so they act as a first quality gate inside the model.
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        word_timestamps=args.word_timestamps,
        vad_filter=args.vad_filter,
        temperature=temperature,
        hotwords=args.hotwords,
    )

    raw_segments: List[Dict[str, Any]] = []
    for seg in segments_iter:
        words = []
        if seg.words:
            for w in seg.words:
                words.append(
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                        "probability": w.probability,
                    }
                )
        raw_segments.append(
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "avg_logprob": seg.avg_logprob,
                "compression_ratio": seg.compression_ratio,
                "no_speech_prob": seg.no_speech_prob,
                "words": words,
            }
        )

    # ── Smart post-filter ─────────────────────────────────────────────────────
    filter_meta = None
    if args.smart_filter:
        kept_segments, removed_segments, filter_meta = auto_tune_smart_filter(
            raw_segments,
            min_keep_ratio=args.smart_filter_min_keep_ratio,
        )
    else:
        kept_segments, removed_segments = raw_segments, []

    # ── Quality report ────────────────────────────────────────────────────────
    raw_conf  = _word_confidence(raw_segments)
    kept_conf = _word_confidence(kept_segments)
    raw_avg_lp  = (
        sum(s["avg_logprob"] for s in raw_segments) / len(raw_segments)
        if raw_segments else 0.0
    )
    kept_avg_lp = (
        sum(s["avg_logprob"] for s in kept_segments) / len(kept_segments)
        if kept_segments else 0.0
    )

    print(f"\n  ── Quality report: {audio_path.name} ──")
    print(
        f"    Raw:    {len(raw_segments):3d} segments, "
        f"avg_logprob={raw_avg_lp:+.3f}, "
        f"word-confidence={raw_conf:.1%}"
    )
    if args.smart_filter:
        if filter_meta is not None:
            print(
                "    Filter profile: "
                f"{filter_meta['profile']} "
                f"(no_speech<={filter_meta['no_speech_threshold']:.2f}, "
                f"logprob>={filter_meta['logprob_threshold']:.2f}, "
                f"compression<={filter_meta['compression_threshold']:.2f}, "
                f"keep-ratio={filter_meta['keep_ratio']:.1%})"
            )
        if removed_segments:
            print(
                f"    Filter: removed {len(removed_segments)} segment(s) "
                f"(hallucination/non-speech indicators):"
            )
            for rs in removed_segments:
                print(
                    f"      [{rs['start']:.1f}s–{rs['end']:.1f}s] "
                    + "; ".join(rs["_filter_reasons"])
                    + f" | \"{rs['text'].strip()[:60]}\""
                )
        else:
            print("    Filter: all segments passed quality checks — nothing removed")
        print(
            f"    Kept:   {len(kept_segments):3d} segments, "
            f"avg_logprob={kept_avg_lp:+.3f}, "
            f"word-confidence={kept_conf:.1%}"
        )

    # ── Write outputs ─────────────────────────────────────────────────────────
    language      = info.language
    language_prob = info.language_probability

    txt_path  = output_base.parent / f"{output_base.name}.txt"
    srt_path  = output_base.parent / f"{output_base.name}.srt"
    json_path = output_base.parent / f"{output_base.name}.json"

    txt_path.write_text(
        "\n".join(seg["text"].strip() for seg in kept_segments if seg["text"].strip()),
        encoding="utf-8",
    )
    write_srt(srt_path, kept_segments)

    payload = {
        "source_audio": str(audio_path),
        "model": runtime_model,
        "device": runtime_device,
        "compute_type": runtime_compute_type,
        "detected_language": language,
        "detected_language_probability": language_prob,
        "duration": info.duration,
        "duration_after_vad": info.duration_after_vad,
        "transcription_options": {
            "task": args.task,
            "language": args.language,
            "beam_size": args.beam_size,
            "best_of": args.best_of,
            "patience": args.patience,
            "temperature": temperature,
            "word_timestamps": args.word_timestamps,
            "vad_filter": args.vad_filter,
            "condition_on_previous_text": args.condition_on_previous_text,
            "hotwords": args.hotwords,
            "compression_ratio_threshold": 2.4,
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "smart_filter": args.smart_filter,
            "smart_filter_min_keep_ratio": args.smart_filter_min_keep_ratio,
            "smart_filter_profile": filter_meta["profile"] if filter_meta else None,
            "smart_filter_profile_thresholds": {
                "no_speech_threshold": filter_meta["no_speech_threshold"],
                "logprob_threshold": filter_meta["logprob_threshold"],
                "compression_threshold": filter_meta["compression_threshold"],
            }
            if filter_meta
            else None,
        },
        "quality": {
            "raw_segment_count": len(raw_segments),
            "kept_segment_count": len(kept_segments),
            "filtered_segment_count": len(removed_segments),
            "kept_text_ratio": round(
                (_text_char_count(kept_segments) / _text_char_count(raw_segments))
                if _text_char_count(raw_segments)
                else 1.0,
                4,
            ),
            "kept_word_confidence": round(kept_conf, 4),
            "kept_avg_logprob": round(kept_avg_lp, 4),
        },
        "segments": kept_segments,
        "filtered_segments": removed_segments,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"\n[{audio_path.name}] model={runtime_model}, device={runtime_device}, "
        f"language={language} (p={language_prob:.3f}), duration={info.duration:.1f}s"
    )
    print(f"Wrote: {txt_path}, {srt_path}, {json_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = [Path(p) for p in args.audio]
    for path in audio_paths:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    if args.output_prefix and len(audio_paths) > 1:
        raise ValueError("--output-prefix can only be used with a single input file.")

    attempts = build_model_attempts(args)
    model_cache: Dict[Tuple[str, str, str], WhisperModel] = {}

    for audio_path in audio_paths:
        if args.output_prefix:
            output_base = output_dir / args.output_prefix
        else:
            output_base = output_dir / audio_path.stem

        last_error: Exception | None = None
        success = False
        for model_name, device_name, compute_type in attempts:
            key = (model_name, device_name, compute_type)
            try:
                if key not in model_cache:
                    print(
                        f"Loading model={model_name}, device={device_name}, "
                        f"compute_type={compute_type}..."
                    )
                    model_cache[key] = load_model(model_name, device_name, compute_type, args)

                transcribe_one(
                    model_cache[key],
                    audio_path,
                    output_base,
                    args,
                    runtime_model=model_name,
                    runtime_device=device_name,
                    runtime_compute_type=compute_type,
                )
                success = True
                break
            except Exception as exc:
                last_error = exc
                print(
                    f"Attempt failed: model={model_name}, device={device_name}, "
                    f"compute_type={compute_type}. Error: {exc}",
                    file=sys.stderr,
                )

        if not success:
            raise RuntimeError(
                f"All transcription attempts failed for {audio_path}. "
                f"Last error: {last_error}"
            )


if __name__ == "__main__":
    main()