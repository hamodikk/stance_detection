import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the file with both manual and LLM annotations
df = pd.read_csv("annotation_comparison.csv")

# Clean up the labels
df['Stance'] = df['Stance'].str.strip().str.lower()
df['llm_predicted_stance'] = df['llm_predicted_stance'].str.strip().str.lower()

# Drop rows with missing labels
valid_labels = ["support", "oppose", "neutral"]
df = df[df['Stance'].isin(valid_labels) & df['llm_predicted_stance'].isin(valid_labels)]

# Extract the labels
y_manual = df['Stance']
y_llm = df['llm_predicted_stance']

# Confusion matrix
labels = valid_labels
cm = confusion_matrix(y_manual, y_llm, labels=labels)

# Plot the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='YlGnBu',
            xticklabels=labels,
            yticklabels=labels)
plt.xlabel('LLM Predicted')
plt.ylabel('Manual Annotated')
plt.title('Confusion Matrix: LLM vs Manual Annotation')
plt.tight_layout()
plt.show()

# Classification report
print("Classification Report:")
print(classification_report(y_manual, y_llm))

# Show label agreement using Cohen's Kappa
kappa = cohen_kappa_score(y_manual, y_llm, labels=labels)
print(f"\nCohen's Kappa: {kappa:.3f}")