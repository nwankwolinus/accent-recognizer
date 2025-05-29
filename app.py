import streamlit as st
import os
import torchaudio
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import numpy as np

# ---- CONFIG ----
REFERENCE_FILES = {
    "american": ["american_1.wav", "american_2.wav", "american_3.wav"],
    "british": ["british_1.wav", "british_2.wav", "british_3.wav"],
    "indian": ["indian_1.wav", "indian_2.wav", "indian_full.wav"],
    "nigerian": ["nigerian_1.wav", "nigerian_2.wav", "nigerian_full.wav"],
}

# ---- MODEL LOADING ----
@st.cache_resource
def get_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

processor, model = get_model()

# ---- EMBEDDING UTILS ----
def preprocess_wav(src, dst):
    waveform, sr = torchaudio.load(src)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    torchaudio.save(dst, waveform, 16000)

def extract_short_embedding(file_path, duration_sec=10):
    waveform, sample_rate = torchaudio.load(file_path)
    max_samples = int(sample_rate * duration_sec)
    waveform = waveform[:, :max_samples]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---- REFERENCE EMBEDDING LOADING ----
@st.cache_resource
def build_reference_embeddings():
    ref_embeds = {}
    load_log = []
    for accent, files in REFERENCE_FILES.items():
        ref_embeds[accent] = []
        for file in files:
            if not os.path.isfile(file):
                load_log.append(f"❌ Missing file: {file}")
                continue
            try:
                emb = extract_short_embedding(file, duration_sec=10)
                ref_embeds[accent].append(emb)
                load_log.append(f"✅ Loaded: {file}")
            except Exception as e:
                load_log.append(f"❌ Error processing {file}: {e}")
    return ref_embeds, load_log

reference_embeddings, reference_log = build_reference_embeddings()

# ---- STREAMLIT UI ----
st.title("Accent Recognition from YouTube Video")
st.write("Paste a YouTube video URL below. The app will extract audio and predict the accent.")

st.subheader("Reference File Status")
for line in reference_log:
    if line.startswith("✅"):
        st.success(line)
    else:
        st.error(line)

video_url = st.text_input("YouTube video URL:")

if st.button("Analyze Accent"):
    if not video_url.strip():
        st.warning("Please enter a YouTube URL.")
        st.stop()

    with st.spinner("Downloading and processing audio..."):
        # Download audio with yt-dlp
        os.system(f'yt-dlp -x --audio-format wav -o "input_audio.%(ext)s" "{video_url}"')

        # Ensure file exists
        if not os.path.isfile("input_audio.wav"):
            st.error("Audio download failed. Make sure yt-dlp is installed and URL is valid.")
            st.stop()
        preprocess_wav('input_audio.wav', 'input_audio.wav')

        # Extract embedding from input audio
        try:
            input_emb = extract_short_embedding('input_audio.wav', duration_sec=10)
        except Exception as e:
            st.error(f"Error processing input audio: {e}")
            st.stop()

        # Compute similarities
        scores = {}
        for accent, embs in reference_embeddings.items():
            similarities = [cosine_similarity(input_emb, ref_emb) for ref_emb in embs if ref_emb is not None]
            if len(similarities) > 0:
                scores[accent] = float(np.mean(similarities))
            else:
                scores[accent] = float('-inf')

        if all(v == float('-inf') for v in scores.values()):
            st.error("No valid reference embeddings found. Check your reference files.")
        else:
            predicted_accent = max(scores, key=scores.get)
            st.success(f"Predicted accent: {predicted_accent}")
            st.write("Similarity scores:")
            st.json(scores)
