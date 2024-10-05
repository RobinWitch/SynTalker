import torch
from transformers import pipeline
import librosa
import os
os.environ["http_proxy"] = "http://10.76.5.191:7890"
os.environ["https_proxy"] = "http://10.76.5.191:7890"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
  device=device,
)

audio,sr = librosa.load("/mnt/data3/cbh/SynTalker/demo/test3/1_wayne_0_2_2.wav",sr=None)
sample = audio

prediction = pipe(sample.copy(), batch_size=8)["text"]

# # we can also return timestamps for the predictions
# prediction = pipe(sample.copy(), batch_size=8, return_timestamps=True)["chunks"]
