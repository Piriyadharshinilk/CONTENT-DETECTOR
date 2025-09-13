import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ---------------- Load Fine-Tuned Model ----------------
@st.cache_resource
def load_text_model():
    model_path = "./artifacts/text_model"   # Path where train.py saved the model
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.eval()  # Important: set model to evaluation mode
    return tokenizer, model

tokenizer, text_model = load_text_model()

# ---------------- Streamlit UI ----------------
st.title("🛡 AI vs Human Text Detector")

user_text = st.text_area("Paste your text here")

if st.button("Check Text"):
    try:
        inputs = tokenizer(user_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            outputs = text_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs).item()
            confidence = probs[0][pred_class].item()

        label = "AI-Generated" if pred_class == 1 else "Human-Written"
        st.success(f"Prediction: {label} (Confidence: {confidence:.2f})")

    except Exception as e:
        st.error(f"Text model error: {e}")
