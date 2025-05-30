import streamlit as st
import os
import soundfile as sf
import requests
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

@st.cache_resource
def get_model():
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    return processor, model

processor, model = get_model()

def preprocess_wav(src, dst):
    import torchaudio
    waveform, sr = torchaudio.load(src)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    torchaudio.save(dst, waveform, 16000)

def extract_short_embedding(file_path, duration_sec=10, use_soundfile=False):
    if use_soundfile:
        signal, sample_rate = sf.read(file_path, dtype='float32')
        if signal.ndim == 1:
            waveform = signal[np.newaxis, :]
        else:
            waveform = signal.T
        waveform = waveform[:, :int(sample_rate * duration_sec)]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(axis=0, keepdims=True)
        if sample_rate != 16000:
            import torchaudio
            waveform_tensor = torch.from_numpy(waveform).float()
            waveform_tensor = torchaudio.transforms.Resample(sample_rate, 16000)(waveform_tensor)
            waveform = waveform_tensor.numpy()
        else:
            waveform = waveform.astype(np.float32)
    else:
        import torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        max_samples = int(sample_rate * duration_sec)
        waveform = waveform[:, :max_samples]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        waveform = waveform.numpy().astype(np.float32)
    waveform_tensor = torch.from_numpy(waveform).float()
    inputs = processor(waveform_tensor.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

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
                emb = extract_short_embedding(file, duration_sec=10, use_soundfile=True)
                ref_embeds[accent].append(emb)
                load_log.append(f"✅ Loaded: {file}")
            except Exception as e:
                load_log.append(f"❌ Error processing {file}: {e}")
    return ref_embeds, load_log

reference_embeddings, reference_log = build_reference_embeddings()

def download_file(url, target_path):
    try:
        r = requests.get(url, stream=True, timeout=20)
        r.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return True, None
    except Exception as e:
        return False, str(e)

st.title("Accent Recognition from Audio/Video URL or File")
st.write("Enter a direct audio/video file URL (ending with .wav, .mp3, .mp4, etc.) or upload a file below.")

st.subheader("Reference File Status")
for line in reference_log:
    if line.startswith("✅"):
        st.success(line)
    else:
        st.error(line)

video_url = st.text_input("Audio/Video file URL:")
uploaded_file = st.file_uploader("Or upload an audio/video file (.wav, .mp3, .mp4)", type=["wav", "mp3", "mp4"])

if st.button("Analyze Accent"):
    input_audio_path = None

    if video_url.strip():
        # Check for direct file link
        if any(video_url.lower().endswith(ext) for ext in [".wav", ".mp3", ".mp4"]):
            with st.spinner("Downloading file from URL..."):
                ext = video_url.split('.')[-1]
                input_file = f"input_audio.{ext}"
                success, err = download_file(video_url, input_file)
                if not success:
                    st.error(f"Download failed: {err}")
                    st.stop()
                # If mp3 or mp4, convert to wav for processing
                if ext in ["mp3", "mp4"]:
                    import torchaudio
                    waveform, sr = torchaudio.load(input_file)
                    torchaudio.save("input_audio.wav", waveform, sr)
                    input_audio_path = "input_audio.wav"
                else:
                    input_audio_path = input_file
        else:
            st.error("URL must be a direct link ending with .wav, .mp3, or .mp4")
            st.stop()
    elif uploaded_file is not None:
        with st.spinner("Processing uploaded file..."):
            ext = uploaded_file.name.split('.')[-1]
            audio_path = f"uploaded_audio.{ext}"
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            if ext in ["mp3", "mp4"]:
                import torchaudio
                waveform, sr = torchaudio.load(audio_path)
                torchaudio.save("input_audio.wav", waveform, sr)
                input_audio_path = "input_audio.wav"
            else:
                input_audio_path = audio_path
    else:
        st.warning("Please provide a valid URL or upload a file.")
        st.stop()

    preprocess_wav(input_audio_path, "input_audio.wav")
    try:
        input_emb = extract_short_embedding("input_audio.wav", duration_sec=10)
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        st.stop()

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
