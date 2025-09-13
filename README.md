
# 🕵️‍♂️ AI Content Detector App — Current Status

This is a Streamlit app that detects whether supplied text, images, or short videos are AI-generated or real. The README below has been updated to reflect the current implementation and development status.

---

## ✅ Current project status (short)
- Backend code: implemented in `app.py` — text, image and video flows are present.
- `artifacts/` directory: created in the repo.
- Text model artifact: `artifacts/text_model.joblib` — present (created by `tools/train_text_pipeline.py`).
- Image model artifact: `artifacts/image_model.h5` — NOT present. Image/video modes will show an error until this file is added.
- `requirements.txt`: repaired and re-saved as UTF-8 (minimal dependency set). Please review before full install.
- Added: `tools/train_text_pipeline.py` — trains a minimal Tfidf+LogisticRegression pipeline and saves `artifacts/text_model.joblib` for local testing.
- `app.py` improvements: safer model loading, cached image loader, temp-file cleanup, and friendlier error messages.

---

## 📂 Project structure (relevant files)
```
./
├─ app.py                     # Streamlit app (backend + UI)
├─ requirements.txt           # Cleaned minimal dependencies (UTF-8)
├─ README.md                  # This file (updated)
├─ artifacts/                 # Model artifacts (text model present)
│  └─ text_model.joblib
├─ data/
│  └─ sample.csv              # optional dataset used by training script
├─ tools/
│  └─ train_text_pipeline.py  # trains and saves text_model.joblib
├─ utils/
│  ├─ preprocessing.py
│  └─ functions.py
└─ notebooks/
    └─ model_training.ipynb
```

---

## ⚙️ How the backend currently works
- On import, `app.py` tries to load `artifacts/text_model.joblib` using `joblib`. If missing or load fails, a safe `DummyClassifier` fallback is used.
- Image model loading is lazy and cached via `@st.cache_resource` (`app.get_image_model`) and expects `artifacts/image_model.h5`.
- Image/video flows write uploads to temporary files, sample frames (for video), preprocess to 224x224, normalize to [0,1], and call the Keras model for predictions.
- Text flow expects raw text and uses the saved Tfidf+LogisticRegression pipeline (or DummyClassifier) to produce a label and, when available, a confidence score.

---

## ▶️ How to run locally (Windows PowerShell)
1. (Optional) Activate your venv if you have one:
```powershell
.\venv\Scripts\Activate
```
2. Install dependencies:
```powershell
pip install -r .\requirements.txt
```
3. (Optional) Recreate text model (if you want a fresh local model):
```powershell
python .\tools\train_text_pipeline.py
```
4. Run the Streamlit app:
```powershell
streamlit run .\app.py
```

Notes: if you don't provide `artifacts/image_model.h5`, the image and video checks will display a "Train it first" / "Image model not found" message.

---

## 🔧 What I changed while finishing the backend
- Created `artifacts/` and produced `artifacts/text_model.joblib` using `tools/train_text_pipeline.py`.
- Repaired `requirements.txt` (UTF-8 minimal list) to make pip installs reliable.
- Hardened `app.py` with:
   - safe `joblib` loading for text pipeline with fallback
   - cached image loader via `@st.cache_resource`
   - try/except around predictions with user-friendly errors
   - proper temp-file cleanup after image/video processing

---

## �️ Recommended next actions
1. Add or train an image model and place it at `artifacts/image_model.h5` so image/video modes can run.
2. Review and expand `requirements.txt` to include any additional packages you need (transformers, torch, etc.) before full deployment.
3. Add simple unit tests for `utils/functions.py` and `tools/train_text_pipeline.py`.
4. (Optional) Add more robust upload limits and progress feedback in the UI.

---

If you want, I can: create a tiny placeholder Keras image model for testing, add caching/spinner UI improvements in `app.py`, or run the Streamlit server here to smoke-test (tell me which).
