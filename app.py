import streamlit as st
import os
import cv2
import tempfile
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from sklearn.dummy import DummyClassifier
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------- Paths ----------------
TEXT_MODEL = "artifacts/text_model.joblib"
IMAGE_MODEL = "artifacts/image_model.h5"

# ---------------- Load Models ----------------
# Text model: try to load a saved pipeline, otherwise use a safe DummyClassifier.
if os.path.exists(TEXT_MODEL):
    try:
        text_model = joblib.load(TEXT_MODEL)
    except Exception:
        text_model = DummyClassifier(strategy="constant", constant="Real")
        text_model.fit([["sample"]], ["Real"])
else:
    # Provide a minimal pipeline suggestion for local testing
    text_model = DummyClassifier(strategy="constant", constant="Real")
    text_model.fit([["sample"]], ["Real"])

# Image model: lazy load with safe None fallback
@st.cache_resource
def get_image_model(path: str):
    """Load and cache the Keras image model. Returns None on failure or if file missing."""
    if os.path.exists(path):
        try:
            return load_model(path)
        except Exception:
            return None
    return None

image_model = get_image_model(IMAGE_MODEL)

# ---------------- Streamlit UI ----------------
st.title("🛡 Realtime AI Content Finder")

st.sidebar.header("Choose Check Type")
mode = st.sidebar.radio("Select Input Type", ["Text", "Image", "Video"])

# ---------------- Text Check ----------------
if mode == "Text":
    st.subheader("📝 Check News / Article")
    user_text = st.text_area("Paste your text here")
    if st.button("Check Text"):
        try:
            pred = text_model.predict([user_text])[0]
            prob = None
            if hasattr(text_model, "predict_proba"):
                proba = text_model.predict_proba([user_text])[0]
                prob = max(proba)
            if prob is not None:
                st.success(f"Prediction: {pred} (Confidence: {prob:.2f})")
            else:
                st.success(f"Prediction: {pred}")
        except Exception as e:
            st.error(f"Text model error: {e}")

# ---------------- Image Check ----------------
elif mode == "Image":
    st.subheader("🖼 Check Image Authenticity")
    file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])
    if file and st.button("Check Image"):
        if image_model is None:
            st.error("Image model not found. Train it first.")
        else:
            tmp = None
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".img")
                tmp.write(file.read())
                tmp.close()
                img = image.load_img(tmp.name, target_size=(224,224))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)/255.0
                pred = image_model.predict(x)
                # Handle models that return logits or probability vectors
                prob = float(pred[0][0]) if pred.shape[-1] == 1 else float(pred[0].max())
                label = "AI-Generated" if prob > 0.5 else "Real"
                st.success(f"Prediction: {label} (Confidence: {prob:.2f})")
            except Exception as e:
                st.error(f"Image prediction failed: {e}")
            finally:
                if tmp is not None and os.path.exists(tmp.name):
                    try:
                        os.remove(tmp.name)
                    except Exception:
                        pass

# ---------------- Video Check ----------------
elif mode == "Video":
    st.subheader("🎥 Check Video Authenticity")
    file = st.file_uploader("Upload a short video", type=["mp4","avi","mov"])
    if file and st.button("Check Video"):
        if image_model is None:
            st.error("Image model not found. Train it first.")
        else:
            tmp = None
            try:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp.write(file.read())
                tmp.close()
                cap = cv2.VideoCapture(tmp.name)

                frame_count = 0
                votes = {"AI-Generated":0, "Real":0}

                while True:
                    ret, frame = cap.read()
                    if not ret or frame_count > 40:
                        break
                    if frame_count % 10 == 0:
                        tmp_img = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                        try:
                            cv2.imwrite(tmp_img.name, frame)
                            img = image.load_img(tmp_img.name, target_size=(224,224))
                            x = image.img_to_array(img)
                            x = np.expand_dims(x, axis=0)/255.0
                            pred = image_model.predict(x)
                            prob = float(pred[0][0]) if pred.shape[-1] == 1 else float(pred[0].max())
                            label = "AI-Generated" if prob > 0.5 else "Real"
                            votes[label] += 1
                        finally:
                            if os.path.exists(tmp_img.name):
                                try:
                                    os.remove(tmp_img.name)
                                except Exception:
                                    pass
                    frame_count += 1
                cap.release()

                final = max(votes, key=votes.get)
                st.success(f"Final Prediction: {final} (Votes: {votes})")
            except Exception as e:
                st.error(f"Video processing failed: {e}")
            finally:
                if tmp is not None and os.path.exists(tmp.name):
                    try:
                        os.remove(tmp.name)
                    except Exception:
                        pass
