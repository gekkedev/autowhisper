#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

from faster_whisper import WhisperModel


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
        description="Offline transcription with faster-whisper. Writes .txt, .srt and .json outputs."
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
        help="Number of workers used by faster-whisper (default: 1, compatibility-friendly).",
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
    parser.add_argument("--patience", type=float, default=1.5, help="Decoding patience.")
    parser.add_argument(
        "--temperature",
        default="0.0,0.2,0.4",
        help="Comma-separated temperature schedule.",
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
        default=False,
        help="Enable voice activity detection filter.",
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
        repetition_penalty=1.02,
        no_repeat_ngram_size=3,
        condition_on_previous_text=True,
        log_progress=args.log_progress,
        compression_ratio_threshold=2.2,
        log_prob_threshold=-0.8,
        no_speech_threshold=0.4,
        word_timestamps=args.word_timestamps,
        vad_filter=args.vad_filter,
        temperature=temperature,
        hotwords=args.hotwords,
    )

    segments: List[Dict[str, Any]] = []
    full_text_parts: List[str] = []

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

        segment_obj = {
            "id": seg.id,
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "avg_logprob": seg.avg_logprob,
            "compression_ratio": seg.compression_ratio,
            "no_speech_prob": seg.no_speech_prob,
            "words": words,
        }
        segments.append(segment_obj)
        full_text_parts.append(seg.text.strip())

    language = info.language
    language_prob = info.language_probability

    txt_path = output_base.parent / f"{output_base.name}.txt"
    srt_path = output_base.parent / f"{output_base.name}.srt"
    json_path = output_base.parent / f"{output_base.name}.json"

    txt_path.write_text("\n".join(part for part in full_text_parts if part), encoding="utf-8")
    write_srt(srt_path, segments)

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
            "word_timestamps": args.word_timestamps,
            "vad_filter": args.vad_filter,
            "hotwords": args.hotwords,
            "temperature": temperature,
            "compression_ratio_threshold": 2.2,
            "log_prob_threshold": -0.8,
            "no_speech_threshold": 0.4,
        },
        "segments": segments,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[{audio_path.name}] model={runtime_model}, device={runtime_device}, "
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
