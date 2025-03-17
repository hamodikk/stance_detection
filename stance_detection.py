# ------------------------
# --- Stance Detection ---
# ------------------------

# In this script, we train RoBERTa for stance detection using
# the annotated subset.
# We use the Hugging Face Transformers library to fine-tune
# RoBERTa on the annotated data.
# The model is then evaluated using accuracy, F1 score and
# confusion matrix.

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset, ClassLabel

# Load the annotated data
df = pd.read_csv("manual_annotated_subset.csv") # Change this based on annotated file

# This section is more useful for larger datasets where
# annotation errors may occur.

# Drop rows with missing labels
df = df.dropna(subset=["Stance"])

# Normalize labels
df["Stance"] = df["Stance"].str.strip().str.lower()
label_classes = ["support", "oppose", "neutral"]
label2id = {label: i for i, label in enumerate(label_classes)}
id2label = {i: label for label, i in label2id.items()}
df = df[df["Stance"].isin(label2id)]
df["label"] = df["Stance"].map(label2id)

# Train-test split
train_df, test_df = train_test_split(
    df, test_size=0.4, stratify=df["label"], random_state=42
)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df[['Text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['Text', 'label']])

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize(example):
    return tokenizer(example['Text'], truncation=True)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

# Add label label information to the dataset
features = train_dataset.features.copy()
features['label'] = ClassLabel(num_classes=3, names=label_classes)
train_dataset = train_dataset.cast(features)
test_dataset = test_dataset.cast(features)

# Load the model
model = RobertaForSequenceClassification.from_pretrained('roberta-base',
                                                         num_labels=3)

# Training configuration
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=6,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
)

# Data collator and trainer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train the model
preds_output = trainer.predict(test_dataset)
preds = preds_output.predictions.argmax(axis=1)
true_labels = test_dataset['label']

print("Classification Report:", classification_report(true_labels,
                                                      preds,
                                                      target_names=label_classes,
                                                      zero_division=1))
print("Confusion Matrix:", confusion_matrix(true_labels, preds))

cm = confusion_matrix(true_labels, preds, labels=[0, 1, 2])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_classes, yticklabels=label_classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Manual Annotated Corpus (15 Samples)")
plt.tight_layout()
plt.savefig("confusion_matrix_subset.png")  # Change filename for other versions
plt.close()

# Save predictions for analysis and compxarison
output_df = test_df.copy()
output_df['predicted_label_id'] = preds
output_df['predicted_stance'] = [id2label[i] for i in preds]

output_df.to_csv("stance_predictions_sample15.csv", index=False) # Change this based on annotated files

with open("classification_report_sample15.txt", "w") as f: # Change this based on annotated file
    f.write(classification_report(true_labels,
                                  preds,
                                  target_names=label_classes,
                                  zero_division=1))