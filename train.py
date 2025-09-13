import os
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import torch

# ---------------- 0. Paths ----------------
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_FILE = os.path.join(PROJECT_ROOT, "ai_human_content_detection_dataset.csv")
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "text_model")

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        f"Dataset not found at {DATA_FILE}. Please place ai_human_content_detection_dataset.csv in the project root."
    )

# ---------------- 1. Load Dataset ----------------
ds = load_dataset("csv", data_files={"train": DATA_FILE})

# ---------------- 2. Initialize Tokenizer ----------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# ---------------- 3. Preprocess Function ----------------
def preprocess(batch):
    # Find a sensible text column in the batch (common names: text_content, text, prompt)
    candidates = ["text_content", "text", "prompt", "content", "text_conte"]
    key = next((k for k in candidates if k in batch), None)
    if key is None:
        # fallback: pick the first column that contains strings
        for k, v in batch.items():
            if isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], str):
                key = k
                break
    if key is None:
        raise KeyError("No text column found in dataset batch; expected 'text_content' or similar.")
    return tokenizer(batch[key], truncation=True, padding="max_length", max_length=256)

# ---------------- 4. Labeling Function ----------------
def add_labels(example):
    # expects a 'label' column in your CSV (already 0 or 1)
    example["label"] = example.get("label")
    return example

# ---------------- 5. Tokenize & Add Labels ----------------
print("Tokenizing dataset...")
train_ds = ds["train"].map(preprocess, batched=True)
train_ds = train_ds.map(add_labels)

# Keep only tensors expected by the model
columns_to_remove = [c for c in train_ds.column_names if c not in ["input_ids", "attention_mask", "label"]]
if columns_to_remove:
    train_ds = train_ds.remove_columns(columns_to_remove)

train_ds.set_format(type="torch")

# ---------------- 6. Load Model ----------------
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# ---------------- 7. Training Arguments ----------------
training_args = TrainingArguments(
    output_dir=ARTIFACT_DIR,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir=os.path.join(PROJECT_ROOT, "logs"),
    logging_steps=10,
)

# ---------------- 8. Trainer ----------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=train_ds,
    tokenizer=tokenizer,
)

# ---------------- 9. Train Model ----------------
print("Starting training...")
trainer.train()

# ---------------- 10. Save Model ----------------
os.makedirs(ARTIFACT_DIR, exist_ok=True)
model.save_pretrained(ARTIFACT_DIR)
tokenizer.save_pretrained(ARTIFACT_DIR)

print(f"Training complete! Model saved in {ARTIFACT_DIR}")
