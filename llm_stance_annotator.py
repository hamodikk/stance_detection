import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# Load the corpus
df = pd.read_csv("full_corpus_annotation_template.csv")

# Initialize zero-shot pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define possible labels
candidate_labels = ["support", "oppose", "neutral"]

# Run the classifier
tqdm.pandas()

def get_llm_stance(text):
    try:
        result = classifier(text, candidate_labels)
        return pd.Series({
            "llm_predicted_stance": result['labels'][0],
            "llm_predicted_score": result['scores'][result['labels'].index('support')],
            "llm_oppose_score": result['scores'][result['labels'].index('oppose')],
            "llm_neutral_score": result['scores'][result['labels'].index('neutral')]
        })
    except:
        return pd.Series({
            "llm_predicted_stance": "error",
            "llm_predicted_score": 0,
            "llm_oppose_score": 0,
            "llm_neutral_score": 0
        })
        
# Apply the function to each row
llm_results = df["Text"].progress_apply(get_llm_stance)

df_with_llm = pd.concat([df, llm_results], axis=1)

# Save the results
df_with_llm.to_csv("llm_stance_predictions.csv", index=False)