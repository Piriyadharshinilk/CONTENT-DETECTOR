# 🔍 AI Content Detector

This project detects whether a given text is **AI-generated** or **human-written** using a fine-tuned **DistilBERT** model from Hugging Face.  
It includes both the **training pipeline** and an **interactive web app** built with **Streamlit**.

---

## ✨ Features
- ✅ Fine-tuned **DistilBERT** for binary text classification  
- ✅ Detects **AI-generated** vs **Human-written** text  
- ✅ Easy-to-use **Streamlit interface**  
- ✅ Reusable model artifacts (no need to retrain every time)  
- ✅ Built with **modern NLP tools (Hugging Face + PyTorch)**  

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Hugging Face Transformers** – Model & training
- **Datasets** – For handling CSV dataset
- **PyTorch** – Backend deep learning framework
- **Streamlit** – Web app for predictions

---

## 📂 Project Structure
CONTENT-DETECTOR/
│
├── artifacts/
│ └── text_model/ # Saved fine-tuned model + tokenizer
│
├── ai_human_content_detection_dataset.csv # Training dataset
│
├── train.py # Model training script
├── app.py # Streamlit web app
├── requirements.txt # Dependencies
└── README.md # Documentation

---

## 📝 File & Folder Explanation

### 1. **artifacts/**
- Stores **trained models** after running `train.py`.  
- Inside `text_model/`, you’ll find:
  - `config.json` → Model architecture details  
  - `model.safetensors` → Trained weights  
  - `tokenizer.json`, `vocab.txt` → Tokenizer files  

> These files are automatically **loaded by app.py** to make predictions.

---

### 2. **ai_human_content_detection_dataset.csv**
- Dataset used for training.  
- Columns:
  - `prompt` → Input text  
  - `human` → Label (`1 = Human-written`, `0 = AI-generated`)

---

### 3. **train.py**
- Handles the **training pipeline**:
  - Loads dataset from CSV  
  - Tokenizes text using DistilBERT tokenizer  
  - Splits into training & test sets  
  - Fine-tunes DistilBERT for binary classification  
  - Saves trained model into `artifacts/text_model/`

---

### 4. **app.py**
- The **web application** built with **Streamlit**.  
- Features:
  - Loads the fine-tuned model from `artifacts/text_model/`  
  - Provides a **text input box** for users  
  - Returns prediction:
    - ✅ **Human-written**  
    - 🚨 **AI-generated**  
  - Displays **confidence score**

---

### 5. **requirements.txt**
Contains dependencies:
🚀 Running the Project
1. Clone Repository
git clone https://github.com/<your-username>/CONTENT-DETECTOR.git
cd CONTENT-DETECTOR

2. Setup Virtual Environment & Install Dependencies
python -m venv venv
venv\Scripts\activate      # Windows
# or
source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt

3. Train Model (Optional)
python train.py

👉 Saves fine-tuned model in artifacts/text_model/
4. Run Web App
streamlit run app.py

🧠 Machine Learning Model
Base model: DistilBERT (distilbert-base-uncased)
Fine-tuned for binary classification
0 → AI-generated
1→ Human-written
The model is lightweight, fast, and suitable for real-time predictions.
