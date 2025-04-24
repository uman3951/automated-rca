import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load the saved vectorizer and LDA model
vectorizer = joblib.load("count_vectorizer.pkl")
best_lda_model = joblib.load("lda_model_tuned.pkl")

# Load the training data
with open('processed_traning_data.json', 'r') as file:
    training_data = json.load(file)

# Convert the training data into a DataFrame
training_df = pd.DataFrame(training_data)

# Transform the 'message' column into feature vectors
X_features = vectorizer.transform(training_df['message'])

# Compute log-likelihood and perplexity for different topic numbers
topic_range = [5, 10, 15, 20, 25]  # Adjust based on tuning
log_likelihoods = []
perplexities = []

for n_topics in topic_range:
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X_features)
    
    log_likelihood = lda.score(X_features)
    perplexity = lda.perplexity(X_features)

    log_likelihoods.append(log_likelihood)
    perplexities.append(perplexity)

    print(f"Topics: {n_topics}, Log-Likelihood: {log_likelihood}, Perplexity: {perplexity}")

# Create two separate plots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# Plot Log-Likelihood
axes[0].plot(topic_range, log_likelihoods, marker="o", linestyle="-", color="blue", label="Log-Likelihood")
axes[0].set_xlabel("Number of Topics")
axes[0].set_ylabel("Log-Likelihood")
axes[0].set_title("LDA Model Log-Likelihood")
axes[0].grid(True)

# Plot Perplexity
axes[1].plot(topic_range, perplexities, marker="s", linestyle="--", color="red", label="Perplexity")
axes[1].set_xlabel("Number of Topics")
axes[1].set_ylabel("Perplexity")
axes[1].set_title("LDA Model Perplexity")
axes[1].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
