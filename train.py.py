"""train.py

Minimal, clean DistilBERT training script.

Expects a CSV named `ai_human_content_detection_dataset.csv` in the project root with at
least these columns:
 - prompt: the text to classify
 - human: boolean-like flag (True/False or 1/0) indicating human-written

The script tokenizes, trains a DistilBERT sequence classification model and saves the
artifact under `artifacts/text_model`.
"""

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
    # Tokenize the 'prompt' column (adjust if your CSV uses a different column name)
    return tokenizer(batch["prompt"], truncation=True, padding="max_length", max_length=256)


# ---------------- 4. Labeling Function ----------------
def add_labels(example):
    # expects a boolean-like 'human' column; map to 0/1
    example["label"] = 0 if example.get("human") else 1
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
