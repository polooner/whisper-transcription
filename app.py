import os
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset
from pytube import YouTube


device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Check for yt video
mp3_file_exists = os.path.exists("mamba.mp3")

# Download yt video if does not exist yet
yt = YouTube("https://www.youtube.com/watch?v=ouF-H35atOY")
title = yt.title
video = (
    yt.streams.filter(only_audio=True)
    .first()
    .download(filename="mamba_a_replacement_for_transformers")
)


dataset = load_dataset(
    "mp3",
    data_files="mamba_a_replacement_for_transformers.mp3",
)
sample = dataset[0]["audio"]

result = pipe(sample, return_timestamps=True)
print(result)
