FROM python:3.10-slim

RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.2.0 torchaudio==2.2.0 "numpy<2" soundfile demucs

WORKDIR /audio
ENTRYPOINT ["demucs"]
