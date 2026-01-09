# Demucs Stem Splitter

A simple Docker wrapper for [Demucs](https://github.com/adefossez/demucs), Meta's audio source separation model.

## Setup (one time)

1. Install [OrbStack](https://orbstack.dev/) (recommended) or [Docker Desktop](https://docker.com/products/docker-desktop)
2. Build the container:
   ```bash
   docker build -t demucs .
   ```

## Usage

### Interactive (recommended)
```bash
./split
```
Lists files in `input/`, lets you pick one, choose stem count, and auto-opens results.

### Quick
```bash
./split "song.mp3" 6
```

### Drag & drop from anywhere
```bash
./split ~/Downloads/song.mp3
```
Auto-copies to `input/` and processes.

## Stem Options

| Mode | Stems | Notes |
|------|-------|-------|
| 6 | vocals, drums, bass, guitar, piano, other | Best quality |
| 4 | vocals, drums, bass, other | Faster |
| vocals | vocals, instrumental | Fastest |

## Output

- Stems saved to `output/<model>/<songname>/`
- Auto-generates `.lof` file - double-click to open all stems in Audacity

## Folder Structure
```
demucs-env/
├── input/      ← drop audio files here
├── output/     ← stems appear here
├── cache/      ← model cache (auto-created)
├── split       ← main script
├── Dockerfile
└── README.md
```
