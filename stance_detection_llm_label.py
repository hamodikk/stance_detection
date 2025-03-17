import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, ClassLabel
import torch

# Load the LLM-labeled corpus
df = pd.read_csv("annotation_comparison.csv")

# Clean labels
df['llm_predicted_stance'] = df['llm_predicted_stance'].str.strip().str.lower()
label_classes = ['support', 'oppose', 'neutral']
label2id = {label: i for i, label in enumerate(label_classes)}
id2label = {i: label for label, i in label2id.items()}

# Filter valid rows and map to label ids
df = df[df['llm_predicted_stance'].isin(label2id)]
df['label'] = df['llm_predicted_stance'].map(label2id)

# Split and tokenize
train_df, test_df = train_test_split(df, test_size=0.4, stratify=df['label'], random_state=42)

train_dataset = Dataset.from_pandas(train_df[['Text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['Text', 'label']])

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(example):
    return tokenizer(example["Text"], truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

features = train_dataset.features.copy()
features['label'] = ClassLabel(num_classes=3, names=label_classes)
train_dataset = train_dataset.cast(features)
test_dataset = test_dataset.cast(features)

# Define model
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

# Training setup
training_args = TrainingArguments(
    output_dir="./results_llm",
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs_llm",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=6,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Evaluate and Save Results
preds_output = trainer.predict(test_dataset)
preds = preds_output.predictions.argmax(axis=1)
true_labels = test_dataset["label"]

# Save predictions for external comparison
test_texts = test_df.copy()
test_texts["llm_model_prediction_id"] = preds
test_texts["llm_model_prediction"] = [id2label[p] for p in preds]
test_texts.to_csv("llm_model_predictions.csv", index=False)

unique = sorted(unique_labels(true_labels, preds))
label_names = [label_classes[i] for i in unique]

# Print evaluation
print("Classification Report (Model Trained on LLM Labels):")
print(classification_report(true_labels, preds, labels=unique, target_names=label_names))
with open("classification_report_llm_labeled.txt", "w") as f:
    f.write(classification_report(true_labels,
                                  preds,
                                  labels=unique,
                                  target_names=label_names
                                  ))

print("Confusion Matrix:")
print(confusion_matrix(true_labels, preds))

cm = confusion_matrix(true_labels, preds, labels=[0, 1, 2])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_classes, yticklabels=label_classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Manual Annotated Corpus")
plt.tight_layout()
plt.savefig("confusion_matrix_llm.png")  # Change filename for other versions
plt.close()