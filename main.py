import streamlit as st
import torch
import torchvision.transforms as transforms
from transformers import ViTConfig, ViTForImageClassification
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt
import joblib
import librosa
import tempfile
import soundfile as sf
import gspread
from google.oauth2.service_account import Credentials
import json
from datetime import datetime

# ========== STYLING ==========
st.set_page_config(page_title="Mood Checker for Seniors", layout="centered")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")

# ========== GOOGLE SHEET FUNCTIONS ==========
def log_to_sheet(input_type, emotion, confidence=None, filename=None):
    try:
        gc = gspread.service_account(filename="creds.json")
        sh = gc.open("mood-detector-log")
        worksheet = sh.sheet1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, input_type, emotion]
        if confidence is not None:
            row.append(f"{confidence:.2f}")
        if filename is not None:
            row.append(filename)
        worksheet.append_row(row)
    except Exception as e:
        st.warning(f"Failed to log to Google Sheets: {e}")

def read_sheet():
    try:
        creds_dict = st.secrets["gcp_service_account"]
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        gc = gspread.authorize(credentials)
        sh = gc.open("mood-detector-log")
        worksheet = sh.sheet1
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.warning(f"Gagal membaca Google Sheets: {e}")
        return pd.DataFrame()

# ========== MODEL DEFINITIONS ==========
class AudioCNN(torch.nn.Module):
    def __init__(self, input_len, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.pool1 = torch.nn.MaxPool1d(2)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.pool2 = torch.nn.MaxPool1d(2)
        self.fc1 = torch.nn.Linear((input_len // 4) * 128, 128)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = torch.nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ========== CONSTANTS & UTILITIES ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad']
emoji_map = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Fear": "😨",
    "Neutral": "😐"
}

insight_map = {
    "Happy": "Great to see a smile today! Keep spreading that positive energy 😊",
    "Sad": "It’s okay to feel down sometimes. Try doing something you enjoy or talk to someone you trust 💬",
    "Angry": "Take a deep breath. A short walk or a glass of water might help calm the mind 🚶‍♂️🧘",
    "Fear": "You are not alone. If something worries you, consider reaching out to a loved one 🤝",
    "Neutral": "A calm state is a great place to be. Stay balanced and keep taking care of yourself 🌿"
}

vis_mood_insights = {
    "Happy": (
        "You’ve been in a joyful mood lately, which is wonderful to see! "
        "Moments of happiness often come from simple things — a warm conversation, a morning walk, or a favorite song. "
        "Keep nurturing these joyful moments by staying connected with loved ones, engaging in activities you enjoy, "
        "and taking time to appreciate the present. Your positive mood can be contagious and uplifting to those around you."
    ),
    "Sad": (
        "Lately, you've experienced a pattern of sadness. It's okay — emotions like sadness are a natural part of life. "
        "They often signal the need for rest, reflection, or emotional connection. "
        "Consider reaching out to someone you trust, journaling your feelings, or spending time doing things that bring you comfort. "
        "Be gentle with yourself and allow space to heal without pressure."
    ),
    "Angry": (
        "Frequent moments of anger have been observed. Anger can arise from stress, unmet needs, or frustration. "
        "While it's a valid emotion, holding onto it too long can be draining. "
        "Try exploring ways to release anger in healthy ways — such as deep breathing, mindful walking, or creative outlets like drawing or music. "
        "Also, reflect on what might be triggering this emotion so you can address it calmly and constructively."
    ),
    "Fear": (
        "There seems to be a recurring presence of fear or anxiety. This might indicate uncertainty or concerns that weigh on your mind. "
        "You don’t have to face them alone. Building routines, limiting overwhelming inputs, and practicing grounding techniques like deep breathing "
        "can help bring a sense of safety. Don’t hesitate to seek support — emotionally, socially, or professionally."
    ),
    "Neutral": (
        "Your emotional state has been largely neutral and stable. This suggests a sense of emotional balance — but also might mean you’re in a routine that lacks stimulation. "
        "Consider introducing small positive changes: try a new hobby, connect with nature, or set light goals for the week. "
        "Emotional stability is a strong foundation, and adding small sparks of joy can brighten each day."
    )
}

# ========== MODEL LOADING ==========
config = ViTConfig.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=len(class_names))
model = ViTForImageClassification(config)
model.load_state_dict(torch.load("models/model_image.pt", map_location=device), strict=False)
model.to(device)
model.eval()

model_audio = AudioCNN(input_len=54, num_classes=len(class_names))
model_audio.load_state_dict(torch.load("models/model_audio.pt", map_location=device))
model_audio.eval()

scaler = joblib.load("models/scaler_audio.pkl")
label_encoder = joblib.load("models/label_encoder_audio.pkl")
label_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}

# ========== PREDICTION FUNCTIONS ==========
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    return transform(image).unsqueeze(0).to(device)

def predict_image(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100
    return class_names[pred_idx], confidence, probs[0].cpu().numpy()

def extract_audio_features(y, sr):
    zcr = librosa.feature.zero_crossing_rate(y)
    rmse = librosa.feature.rms(y=y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=128)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features = np.hstack((
        np.mean(zcr), np.std(zcr),
        np.mean(rmse), np.std(rmse),
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1)
    ))
    return features

def predict_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    features = extract_audio_features(y, sr)
    features = scaler.transform([features])[0]
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model_audio(x)
        probs = torch.softmax(output, dim=1) 
        predicted = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted].item() * 100
    return label_mapping[predicted], confidence, probs[0].numpy(), y, sr

def ensemble_with_confidence(prob_img, conf_img, prob_audio, conf_audio, alpha=0.5, threshold=60):
    combined = alpha * np.array(prob_img) + (1 - alpha) * np.array(prob_audio)
    pred_idx = np.argmax(combined)
    conf_combined = combined[pred_idx] * 100
    final_label = class_names[pred_idx]

    label_img = class_names[np.argmax(prob_img)]
    label_audio = class_names[np.argmax(prob_audio)]

    if conf_combined < threshold and final_label not in [label_img, label_audio]:
        if conf_img > conf_audio:
            return label_img, conf_img
        else:
            return label_audio, conf_audio

    return final_label, conf_combined

# ========== SIDEBAR ==========
st.sidebar.image("assets/logo.jpg", width=80)

st.sidebar.markdown('<div class="sidebar-title">Elderly Mood Detector</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-desc">A simple tool to detect mood using face or voice.</div>', unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "🏠 Landing Page"

if st.sidebar.button("🏠 Landing Page"):
    st.session_state.page = "🏠 Landing Page"

if st.sidebar.button("💗 Mood Detection"):
    st.session_state.page = "💗 Mood Detection"

if st.sidebar.button("📊 Visualization"):
    st.session_state.page = "📊 Visualization"

page = st.session_state.page

# ========== LANDING PAGE ==========
if page == "🏠 Landing Page":
    st.markdown('<p class="big-title">Welcome to Elderly Mood Detector 👵🧓</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("assets/landing_img.jpg", width=400)
    with col2:
        st.markdown("""
        <div class="info-box">
        This smart and friendly tool helps you detect <b>elderly emotions</b> easily.  
        Just upload a <b>face photo</b> or a <b>voice recording</b>, and we’ll tell you how they feel!
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📌 How to Use:")
    st.markdown("""
    <div class="card">
        <ol class="how-to-use">
            <li>Click <b>Mood Detection</b> on the sidebar</li>
            <li>Upload a clear photo or a WAV voice file</li>
            <li>Get the detected mood in just seconds!</li>
            <li>Visit the <b>Visualization</b> tab to track your mood history and insights</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("💡 Emotions We Can Detect:")
    st.markdown("""
    <div class="emotion-box">
        😊 Happy &nbsp;&nbsp;&nbsp; 😢 Sad &nbsp;&nbsp;&nbsp; 😠 Angry &nbsp;&nbsp;&nbsp; 😨 Fear &nbsp;&nbsp;&nbsp; 😐 Neutral
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎞️ How it Works")
    st.markdown("""
    <div class="card">
        🔍 We use a powerful <b>Vision Transformer (ViT)</b> to analyze facial expressions.  
        For voices, we extract features with <b>MFCC & Chroma</b> and use a <b>1D CNN</b> to detect emotions.  
        All processed locally, and results shown immediately!
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p class='medium-font'>Here's how it feels in action:</p>", unsafe_allow_html=True)
    st.video("assets/tutorial.mp4", format="video/mp4", start_time=0, width=400)


# ========== TRY THE MODEL ==========
elif page == "💗 Mood Detection":
    st.markdown('<p class="big-title">💖 How Are Your Feeling Today?</p>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📸 Photo", "🎙️ Voice", "🧠 Ensemble"])

    # Image Tab
    with tab1:
        st.markdown('<p class="big-font">Step 1: Upload a photo</p>', unsafe_allow_html=True)

        image = None
        label = None
        confidence = None

        # Upload image
        uploaded_image = st.file_uploader("Choose a JPG or PNG file", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            image = Image.open(uploaded_image).convert("RGB")
            image_tensor = preprocess_image(image)
            label, confidence, prob_img = predict_image(image_tensor)
            st.session_state['prob_img'] = prob_img
            st.session_state['conf_img'] = confidence

        # Camera capture
        st.markdown('<p class="big-font">Or take a photo using your webcam</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="medium-font justify-text">
        Please position your face clearly in the center of the camera for accurate mood detection.
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            camera_image = st.camera_input("Take a picture")

        if camera_image is not None:
            image = Image.open(camera_image).convert("RGB")
            w, h = image.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            image = image.crop((left, top, left + min_dim, top + min_dim))

            image_tensor = preprocess_image(image)
            label, confidence, prob_img = predict_image(image_tensor)
            st.session_state['prob_img'] = prob_img

        if image and label:
            log_to_sheet("Image", label)
            st.markdown("---")
            st.markdown('<p class="big-font">🎯 Prediction Result</p>', unsafe_allow_html=True)
            st.image(image, caption="🖼️ Image used for prediction", use_container_width=True)
            st.markdown(f"""
            <div class="result-box">
                <div class="emoji">{emoji_map[label]}</div>
                <div class="big-font">Detected Mood: <b>{label}</b></div>
                <div class="medium-font">Confidence: {confidence:.2f}%</div>
                <div class="insight">{insight_map[label]}</div>
            </div>
            """, unsafe_allow_html=True)

    # Audio Tab
    with tab2:
        st.markdown('<p class="big-font">Step 2: Upload a voice recording</p>', unsafe_allow_html=True)
        uploaded_audio = st.file_uploader("Choose a WAV file", type=["wav"])
        st.markdown(
            '<p class="audio-note">'
            '🎤 Don\'t have a WAV file? <a href="https://online-voice-recorder.com/" target="_blank">Record here</a> and upload below.</p>',
            unsafe_allow_html=True
        )
        if uploaded_audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_audio.read())
                temp_audio_path = tmp_file.name
            # Predict
            label_audio, conf_audio, prob_audio, y_audio, sr_audio = predict_audio(temp_audio_path)
            st.session_state['prob_audio'] = prob_audio
            st.session_state['conf_audio'] = conf_audio
            label_audio = str(label_audio).capitalize()
            log_to_sheet("Audio", label_audio)
            # Audio player
            st.audio(temp_audio_path, format="audio/wav")
            st.markdown(f"""
            <div class="result-box">
                <div class="emoji">{emoji_map[label_audio]}</div>
                <div class="big-font">Detected Mood: <b>{label_audio}</b></div>
                <div class="medium-font">Confidence: {conf_audio:.2f}%</div>
                <div class="insight">{insight_map[label_audio]}</div>
            </div>
            """, unsafe_allow_html=True)

    # Ensemble tab
    with tab3:
        st.markdown('<p class="big-font">🧠 Ensemble Mood Prediction</p>', unsafe_allow_html=True)

        if 'prob_img' in st.session_state and 'prob_audio' in st.session_state:
            alpha = st.slider("Adjust Image-Audio Balance", 0.0, 1.0, 0.5, 0.05)

            label_ensemble, conf_ensemble = ensemble_with_confidence(
                st.session_state['prob_img'],
                st.session_state.get('conf_img', 0),
                st.session_state['prob_audio'],
                st.session_state.get('conf_audio', 0),
                alpha=alpha
            )

            st.markdown("#### 🔍 Modality Breakdown")
            st.markdown(f"- 📸 Image: **{class_names[np.argmax(st.session_state['prob_img'])]}** ({st.session_state.get('conf_img', 0):.2f}%)")
            st.markdown(f"- 🎙️ Audio: **{class_names[np.argmax(st.session_state['prob_audio'])]}** ({st.session_state.get('conf_audio', 0):.2f}%)")

            st.markdown(f"""
            <div class="result-box">
                <div class="emoji">{emoji_map[label_ensemble]}</div>
                <div class="big-font">Combined Mood: <b>{label_ensemble}</b></div>
                <div class="medium-font">Based on image & voice, with confidence: {conf_ensemble:.2f}%</div>
                <div class="insight">{insight_map[label_ensemble]}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Please upload both a photo and a voice file first (in the first two tabs) to view the ensemble result.")

# ========== Visualization ==========
elif page == "📊 Visualization":
    st.markdown('<p class="big-title">📊 Mood Trends Overview</p>', unsafe_allow_html=True)
    df = read_sheet()

    if not df.empty:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values(by="Timestamp", ascending=False)

        st.markdown("### 📅 Filter Log")
        filter_option = st.selectbox("Select time range to view mood logs:", ["Latest (7 days)", "Last 1 Month", "All"])

        if filter_option == "Latest (7 days)":
            filtered_df = df[df['Timestamp'] >= pd.Timestamp.now() - pd.Timedelta(days=7)]
        elif filter_option == "Last 1 Month":
            filtered_df = df[df['Timestamp'] >= pd.Timestamp.now() - pd.Timedelta(days=30)]
        else:
            filtered_df = df

        st.markdown(f"Showing **{len(filtered_df)}** mood log(s) based on selected time range.")
        st.dataframe(filtered_df)

        st.subheader("📈 Emotion Count")
        st.markdown("This chart displays the total number of detected emotions based on the selected time range. You can choose between a bar chart or a donut chart for better visual understanding.")

        emotion_counts = filtered_df["Emotion"].value_counts().reset_index()
        emotion_counts.columns = ["Emotion", "Count"]

        emotion_color_map = {
            "Happy": "#FFD700",
            "Sad": "#1F77B4",
            "Angry": "#D62728",
            "Fear": "#800080",
            "Neutral": "#B0B0B0",
        }

        emotion_counts["Color"] = emotion_counts["Emotion"].map(emotion_color_map)

        chart_type = st.selectbox("Select chart type:", ["Bar Chart", "Donut Chart"])

        if chart_type == "Bar Chart":
            chart = alt.Chart(emotion_counts).mark_bar().encode(
                x=alt.X("Emotion:N", title="Emotion"),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color("Emotion:N", scale=alt.Scale(domain=list(emotion_color_map.keys()), range=list(emotion_color_map.values()))),
                tooltip=["Emotion", "Count"]
            ).properties(
                width=600,
                height=400
            )
        else:
            chart = alt.Chart(emotion_counts).mark_arc(innerRadius=50).encode(
                theta=alt.Theta("Count:Q", title=""),
                color=alt.Color("Emotion:N", scale=alt.Scale(domain=list(emotion_color_map.keys()), range=list(emotion_color_map.values()))),
                tooltip=["Emotion", "Count"]
            ).properties(
                width=400,
                height=400
            )

        st.altair_chart(chart, use_container_width=True)
        if not emotion_counts.empty:
            top_emotion_row = emotion_counts.sort_values(by="Count", ascending=False).iloc[0]
            top_emotion = top_emotion_row["Emotion"]
            top_count = top_emotion_row["Count"]

            insight_text = vis_mood_insights.get(top_emotion, "Stay mindful of your emotional state.")
            emoji = emoji_map.get(top_emotion, "🙂")

            st.markdown("---")
            st.markdown(f"### 💡 Dominant Mood Insight: **{top_emotion}** ({top_count} logs)")
            st.markdown(f"""
            <div class="result-box">
                <div class="emoji">{emoji}</div>
                <div class="big-font">Dominant Mood: <b>{top_emotion}</b></div>
                <div class="medium-font">{insight_text}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No logs found yet. Start adding data to view mood trends.")
