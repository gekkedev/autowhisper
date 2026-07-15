# Offline Transcription CLI (CPU-first, fallback-friendly)

Small CLI wrapper around `faster-whisper` to transcribe audio/video **offline** into:

- `*.txt`
- `*.srt`
- `*.json` (with segment + word timestamps and confidence)

Designed for normal machines, with compatibility-first defaults:
- `device=cpu`
- `compute_type=int8`
- automatic model fallback: `large-v3 -> medium -> small -> base`
- optional fallback from GPU to CPU if GPU fails

## Requirements

- Python 3.9+
- `faster-whisper`

Install:
```bash
python3 -m pip install --user faster-whisper
```

## Basic usage

```bash
python3 run_offline_transcribe.py "my_audio.m4a"
```

## Useful examples

Transcribe multiple files:
```bash
python3 run_offline_transcribe.py "a.wav" "b.mp3" --output-dir outputs
```

Force CPU-safe settings:
```bash
python3 run_offline_transcribe.py "call.m4a" --device cpu --compute-type int8
```

Higher quality (slower):
```bash
python3 run_offline_transcribe.py "call.m4a" --language en --beam-size 10 --best-of 10
```

Bias recognition for non-dictionary terms:
```bash
python3 run_offline_transcribe.py "call.m4a" --hotwords "Brand name,odd-spelled w0rd"
```

Use a smaller model directly:
```bash
python3 run_offline_transcribe.py "call.m4a" --model small --no-fallback
```

## Notes

- First run may download model weights.
- Output filenames are based on input filename stem.
- JSON stores the *actual* runtime model/device/compute type used.