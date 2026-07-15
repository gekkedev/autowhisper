# Autowhisper

High-quality offline transcription without manually tuning Whisper parameters. Works with consumer laptops, not just high-end GPUs...

Autowhisper is a command-line wrapper around `faster-whisper` that turns audio
into TXT, SRT, and structured JSON files. It automatically detects the spoken
language and uses sensible transcription defaults to produce strong results
with minimal configuration.

```bash
python autowhisper.py recording.mp3
```

No need to manually experiment with beam size, temperature schedules, patience,
VAD, or language-specific settings for every recording - but you can still pass
custom parameters.

## What you get

- Offline transcription to `.txt`, `.srt`, and `.json`
- Language auto-detection when `--language` is omitted
- CPU-first defaults with model fallback
- Optional smart filtering for low-confidence segments
- Per-file quality metadata in JSON

## Entry point

- `autowhisper.py`
- Run it as `python autowhisper.py ...` (or `python3 ...` on systems where that
  is your Python 3 command)

## Requirements

- Python 3.9+
- `faster-whisper`

Install:
```bash
python -m pip install faster-whisper
```

## Examples

Transcribe multiple files:
```bash
python autowhisper.py a.wav b.mp3 --output-dir outputs
```

Force CPU-safe settings:
```bash
python autowhisper.py call.m4a --device cpu --compute-type int8
```

Higher quality (slower):
```bash
python autowhisper.py call.m4a --language en --beam-size 10 --best-of 10
```

Bias recognition for non-dictionary terms:
```bash
python autowhisper.py call.m4a --hotwords "Brand name,odd-spelled w0rd"
```

Use a smaller model directly:
```bash
python autowhisper.py call.m4a --model small --no-fallback
```

## Notes

- First run may download model weights.
- Output filenames are based on input filename stem.
- JSON stores the actual runtime model/device/compute type used.
- `--smart-filter` and `--vad-filter` are on by default; use `--no-smart-filter`
  or `--no-vad-filter` to disable them.