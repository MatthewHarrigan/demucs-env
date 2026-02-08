"""Demucs Web UI - Gradio interface for stem separation."""

import tempfile
from pathlib import Path

import gradio as gr
import torch
from demucs.apply import apply_model
from demucs.audio import AudioFile, save_audio
from demucs.pretrained import get_model
from demucs.separate import load_track

# --- Device detection ---
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# --- Model cache ---
_models = {}

STEM_CONFIGS = {
    "6 stems (vocals, drums, bass, guitar, piano, other)": ("htdemucs_6s", 6),
    "4 stems (vocals, drums, bass, other)": ("htdemucs", 4),
    "2 stems (vocals, instrumental)": ("htdemucs", 2),
}


def get_cached_model(name):
    if name not in _models:
        _models[name] = get_model(name)
    return _models[name]


def download_youtube(url):
    """Download audio from YouTube URL, return path to file."""
    import yt_dlp

    outdir = Path("input")
    outdir.mkdir(exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "0"}],
        "outtmpl": str(outdir / "%(title)s.%(ext)s"),
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "download")
        # yt-dlp converts to mp3 via postprocessor
        out_path = outdir / f"{title}.mp3"
        if out_path.exists():
            return str(out_path)
        # Fallback: find the most recently created file
        files = sorted(outdir.glob("*"), key=lambda f: f.stat().st_mtime, reverse=True)
        if files:
            return str(files[0])
        raise RuntimeError("Download failed - no output file found")


def split_audio(audio_path, stem_choice, progress=gr.Progress(track_tqdm=True)):
    """Core separation logic. Returns list of (filepath, label) tuples."""
    if audio_path is None:
        raise gr.Error("No audio file provided")

    model_name, stem_count = STEM_CONFIGS[stem_choice]
    model = get_cached_model(model_name)

    # Load audio
    wav = load_track(Path(audio_path), model.audio_channels, model.samplerate)

    # Normalize (from demucs separate.py:170-172)
    ref = wav.mean(0)
    wav -= ref.mean()
    wav /= ref.std()

    # Separate
    sources = apply_model(
        model, wav[None], device=DEVICE, shifts=1, split=True, overlap=0.25, progress=True
    )[0]

    # Denormalize (from demucs separate.py:176-177)
    sources *= ref.std()
    sources += ref.mean()

    # Handle two-stem mode
    if stem_count == 2:
        sources_list = list(sources)
        vocals = sources_list.pop(model.sources.index("vocals"))
        instrumental = torch.zeros_like(sources_list[0])
        for s in sources_list:
            instrumental += s
        stem_names = ["vocals", "instrumental"]
        stem_tensors = [vocals, instrumental]
    else:
        stem_names = list(model.sources)
        stem_tensors = list(sources)

    # Save to temp files
    outdir = Path(tempfile.mkdtemp(prefix="demucs_"))
    results = []
    for name, tensor in zip(stem_names, stem_tensors):
        path = outdir / f"{name}.wav"
        save_audio(tensor, str(path), samplerate=model.samplerate)
        results.append(str(path))

    return results


def on_split_upload(audio, stem_choice):
    return _format_outputs(split_audio(audio, stem_choice), stem_choice)


def on_split_youtube(url, stem_choice):
    if not url or not url.strip():
        raise gr.Error("Please enter a YouTube URL")
    audio_path = download_youtube(url.strip())
    return _format_outputs(split_audio(audio_path, stem_choice), stem_choice)


def _format_outputs(paths, stem_choice):
    """Pad paths list to 6 slots, return updates for all audio outputs."""
    _, stem_count = STEM_CONFIGS[stem_choice]
    updates = []
    for i in range(6):
        if i < len(paths):
            updates.append(gr.update(value=paths[i], visible=True))
        else:
            updates.append(gr.update(value=None, visible=False))
    return updates


# --- Gradio UI ---
with gr.Blocks(title="Demucs Stem Splitter") as demo:
    gr.Markdown(f"# Demucs Stem Splitter\nDevice: **{DEVICE.upper()}**")

    stem_choice = gr.Radio(
        choices=list(STEM_CONFIGS.keys()),
        value="6 stems (vocals, drums, bass, guitar, piano, other)",
        label="Stem count",
    )

    with gr.Tabs():
        with gr.Tab("Upload Audio"):
            audio_input = gr.Audio(type="filepath", label="Audio file")
            split_upload_btn = gr.Button("Split", variant="primary")

        with gr.Tab("YouTube URL"):
            url_input = gr.Textbox(label="YouTube URL", placeholder="https://youtube.com/watch?v=...")
            split_yt_btn = gr.Button("Download & Split", variant="primary")

    # 6 output slots (show/hide based on stem count)
    stem_labels = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    outputs = []
    with gr.Row():
        for label in stem_labels[:3]:
            outputs.append(gr.Audio(label=label, visible=True, interactive=False))
    with gr.Row():
        for label in stem_labels[3:]:
            outputs.append(gr.Audio(label=label, visible=True, interactive=False))

    def update_visibility(choice):
        """Show/hide output slots when stem count changes."""
        _, count = STEM_CONFIGS[choice]
        labels_map = {
            6: ["vocals", "drums", "bass", "guitar", "piano", "other"],
            4: ["vocals", "drums", "bass", "other", "", ""],
            2: ["vocals", "instrumental", "", "", "", ""],
        }
        labels = labels_map[count]
        return [gr.update(visible=bool(labels[i]), label=labels[i] if labels[i] else stem_labels[i]) for i in range(6)]

    stem_choice.change(update_visibility, inputs=stem_choice, outputs=outputs)

    split_upload_btn.click(on_split_upload, inputs=[audio_input, stem_choice], outputs=outputs)
    split_yt_btn.click(on_split_youtube, inputs=[url_input, stem_choice], outputs=outputs)


if __name__ == "__main__":
    demo.launch()
