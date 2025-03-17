import pandas as pd
import re
import string
import nltk
import unicodedata
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources if not already downloaded
nltk.download("punkt")
nltk.download("stopwords")

# As I was working through the project, I noticed that there were some
# errors in the dataset. After some research I realized that these might be
# due to binary data accidentally getting inserted. Following is to fix this issue.

with open("Class_Corpus-1.csv", "rb") as f:
    cleaned = f.read().replace(b"\x00", b"")
    
with open("Class_Corpus-1.csv", "wb") as f:
    f.write(cleaned)

# Load the dataset
corpus_df = pd.read_csv("Class_Corpus-1.csv")

# Normalize the documents
# Some documents in our corpus includes artifacts
# (couldn't becomes couldnâ€™t, etc.)
# We will normalize the text to remove these artifacts for easier training.

def normalize_text(text):
    text = unicodedata.normalize("NFKD", text)
    
    # Remove URLs
    text = re.sub(r"\(https?:\/\/.*?\)", "", text, flags=re.DOTALL)
    text = re.sub(r"https?:\/\/\S+", "", text)

    # Based on the artifact found, this issue seems to be
    # related to UTF-8 encoding
    # We will replace the characters with what they are
    # supposed to represent.
    replacements = {
        "â€™": "'",
        "â€“": "-",
        "â€œ": '"',
        "â€": '"',
        "â€˜": "'",
        "â€¦": "...",
        "Ã©": "é",
        "â€": '"',
        "â€¡": "",
        "Â": "",
        "â€”": "-",
        "™": "",
        "©": "",
        "®": "",
        "°": "",
        "â€¢": "-",
        "·": "-",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # Remove any other unwanted characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text


# Preprocessing function
def clean_doc(text):
    tokens = word_tokenize(str(text))
    re_punc = re.compile("[%s]" % re.escape(string.punctuation))
    tokens = [re_punc.sub("", w) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if len(word) > 4]
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    
    if not tokens:
        return "[empty]"
    
    return " ".join(tokens)

# Apply normalization
corpus_df["Text"] = corpus_df["Text"].apply(normalize_text)

# Apply cleaning
corpus_df["Processed_Text"] = corpus_df["Text"].apply(clean_doc)

# Filter out any cleaned empty text
corpus_df = corpus_df[corpus_df['Processed_Text'].str.strip().str.lower() != '[empty]']

# TF-IDF vectorization
vectorizer = TfidfVectorizer(ngram_range=(1, 1))
tfidf_matrix = vectorizer.fit_transform(corpus_df["Processed_Text"])

# Convert to DataFrame for inspection
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out()
)

# Optional: add titles or IDs back
tfidf_df["Title"] = corpus_df["DSI_Title"]

# Preview results
print(tfidf_df.head())

# ----------------------------------
# --- TF-IDF + KMeans Clustering ---
# ----------------------------------

from sklearn.cluster import KMeans

# Set number of clusters
k = 8

# Remove the 'Title' column if it was added
X = tfidf_df.drop(columns=["Title"], errors="ignore")

# Run KMeans
km = KMeans(n_clusters=k, random_state=42)
km.fit(X)

# Cluster labels
clusters = km.labels_

# Assign clusters to original data
corpus_df["TFIDF_Cluster"] = clusters

# Extract top terms per cluster
terms = vectorizer.get_feature_names_out()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

# Print top terms per cluster
for i in range(k):
    print(f"\nCluster {i} top terms:")
    for ind in order_centroids[i, :10]:
        print(f"  {terms[ind]}")

# Optional: summary of documents per cluster
cluster_summary = corpus_df[["DSI_Title", "TFIDF_Cluster", "Processed_Text"]]
print("\nDocument cluster assignments:")
print(cluster_summary.head())

# ---------------------------------
# --- Train Doc2Vec and Cluster ---
# ---------------------------------

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
import pandas as pd

corpus_df["Processed_Tokens"] = corpus_df["Processed_Text"].apply(lambda x: x.split())

# Prepare TaggedDocument format for gensim
tagged_docs = [
    TaggedDocument(words=tokens, tags=[str(i)])
    for i, tokens in enumerate(corpus_df["Processed_Tokens"])
]

# Train Doc2Vec model
doc2vec_model = Doc2Vec(vector_size=50,
                        window=2,
                        min_count=1,
                        workers=4,
                        epochs=40)
doc2vec_model.build_vocab(tagged_docs)
doc2vec_model.train(
    tagged_docs,
    total_examples=doc2vec_model.corpus_count,
    epochs=doc2vec_model.epochs
)

# Infer vectors for each document
doc_vectors = [doc2vec_model.infer_vector(doc.words) for doc in tagged_docs]
doc2vec_df = pd.DataFrame(doc_vectors)

# KMeans clustering on doc2vec vectors
k = 8
km_doc2vec = KMeans(n_clusters=k, random_state=42)
km_doc2vec.fit(doc2vec_df)
doc2vec_clusters = km_doc2vec.labels_

# Assign to corpus DataFrame
corpus_df["Doc2Vec_Cluster"] = doc2vec_clusters

# Show first few cluster assignments
print(corpus_df[["DSI_Title", "TFIDF_Cluster", "Doc2Vec_Cluster"]].head())

# Save document vectors if needed
doc2vec_df.to_csv("doc2vec_vectors.csv", index=False)

# ------------------------------------
# --- Visualizing Doc2Vec Clusters ---
# ------------------------------------

import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import MDS
import numpy as np

# Compute pairwise distance matrix (cosine distance = 1 - similarity)
dist = 1 - cosine_similarity(doc2vec_df)

# Run MDS
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
pos = mds.fit_transform(dist)

# Add MDS coordinates to DataFrame
corpus_df["MDS_X"] = pos[:, 0]
corpus_df["MDS_Y"] = pos[:, 1]

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    corpus_df["MDS_X"],
    corpus_df["MDS_Y"],
    c=corpus_df["Doc2Vec_Cluster"],
    cmap="tab10",
    s=100,
    alpha=0.8,
)
plt.title("Doc2Vec Clusters (MDS Projection)", fontsize=14)
plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.show()


# -----------------------------------
# --- Visualizing TF-IDF Clusters ---
# -----------------------------------

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Drop Title column from tfidf_df if it's there
X_tfidf = tfidf_df.drop(columns=["Title"], errors="ignore")

# Run PCA to reduce to 2D
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(X_tfidf.to_numpy())

# Add PCA results to corpus_df
corpus_df["PCA_X"] = pca_result[:, 0]
corpus_df["PCA_Y"] = pca_result[:, 1]

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    corpus_df["PCA_X"],
    corpus_df["PCA_Y"],
    c=corpus_df["TFIDF_Cluster"],
    cmap="tab10",
    s=100,
    alpha=0.8,
)
plt.title("TF-IDF Clusters (PCA Projection)", fontsize=14)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.show()

# ---------------------------------
# --- Subsetting for Annotation ---
# ---------------------------------

# In this section, we will subset the corpus and manually 
# label them for stance detection.
# This subset will be used to train a supervised model for
# stance detection.

# Sample a subset for annotation
sample_size = 15
annotation_df = (
    corpus_df[["DSI_Title", "Text"]].sample(n=sample_size,
                                            random_state=42).copy()
)

# Add blank columns for manual annotation
annotation_df["Stance"] = ""
# annotation_df['Claim'] = ''
# annotation_df['Premise'] = ''
# annotation_df['Conclusion'] = ''

# Save to CSV for manual labeling
annotation_df.to_csv("manual_annotation_template.csv", index=False)

# Additionally, save a full corpus annotation template
full_corpus_df = corpus_df[["DSI_Title", "Text"]].copy()
full_corpus_df['Stance'] = ''
full_corpus_df.to_csv("full_corpus_annotation_template.csv", index=False)

# We save two different annotation files for model accuracy comparison
# based on corpus size.

# Once we save this template, we manually annotate each text 
# based on their stance on the topic.
# Note that we can optionally add more columns for claim, 
# premise, conclusion, etc. based on the task.

# Next step will be loading the annotated data and train a 
# stance detection model. (stance_detection.py)
