# Demucs Stem Splitter

A wrapper for [Demucs](https://github.com/adefossez/demucs), Meta's audio source separation model.

## Quick Start

```bash
./setup    # Install everything (ffmpeg, Python packages, etc.)
./ui       # Launch web interface at localhost:7860
```

## Web Interface

Run `./ui` to open a Gradio web UI with:
- Upload audio files or paste a YouTube URL
- Choose 6, 4, or 2 stem separation
- Listen to separated stems in the browser
- Automatic MPS/CUDA/CPU device detection

## Setup

### Native (faster, recommended for Mac)
```bash
./setup
# Or manually:
brew install ffmpeg
uv venv && source .venv/bin/activate && uv pip install demucs yt-dlp soundfile gradio
```

### Docker (portable)
```bash
# Install OrbStack or Docker Desktop first
docker build -t demucs .
```

## CLI Usage

```bash
# Download from YouTube
./download "https://youtube.com/watch?v=..."

# Split into stems (interactive)
./split

# Or specify file and mode directly
./split "song.mp3" 6
```

Use `--docker` flag for portability (slower, but works anywhere):
```bash
./download --docker "https://youtube.com/watch?v=..."
./split --docker
```

Features:
- List files in `input/`, pick one, choose stem count
- Auto-open results in Finder
- Generate `.lof` file for Audacity

## Stem Options

| Mode | Stems | Notes |
|------|-------|-------|
| 6 | vocals, drums, bass, guitar, piano, other | Best quality |
| 4 | vocals, drums, bass, other | Faster |
| vocals | vocals, instrumental | Fastest |

## Output

- Stems saved to `output/<model>/<songname>/`
- Double-click `.lof` file to open all stems in Audacity
