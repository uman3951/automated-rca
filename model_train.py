import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib

# Load the training data
with open('processed_traning_data.json', 'r') as file:
    training_data = json.load(file)

# Convert the training data into a DataFrame for processing
training_df = pd.DataFrame(training_data)

# Verify if 'message' and 'root_cause' columns exist in the training data
if 'message' not in training_df.columns or 'root_cause' not in training_df.columns:
    raise ValueError("The columns 'message' or 'root_cause' do not exist in the training data.")

# Fill missing values in 'root_cause'
training_df['root_cause'] = training_df['root_cause'].fillna("Unknown Issue")

# Get unique root causes and create a mapping
unique_root_causes = training_df['root_cause'].unique()
root_cause_mapping = {i: cause for i, cause in enumerate(unique_root_causes)}

print("Root Cause Mapping:", root_cause_mapping)

# Initialize the CountVectorizer
vectorizer = CountVectorizer(max_features=1000, stop_words='english')

# Fit and transform the 'message' column into features
X_features = vectorizer.fit_transform(training_df['message'])

# Initialize the LDA model
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)

# Fit the LDA model to the features
lda_model.fit(X_features)

# Save the vectorizer and the trained LDA model
joblib.dump(vectorizer, "count_vectorizer.pkl")
joblib.dump(lda_model, "lda_model.pkl")

print("Training complete. Models saved as 'count_vectorizer.pkl' and 'lda_model.pkl'.")

# Assign topics (or root causes) to the training data
topic_probabilities = lda_model.transform(X_features)
training_df['topic_index'] = topic_probabilities.argmax(axis=1)
training_df['root_cause'] = training_df['topic_index'].map(root_cause_mapping)


# # Drop the topic index column if not needed
# training_df.drop(columns=['topic_index'], inplace=True)

# # Save the training data with root causes back to a file
# with open('pre_processed_logs_with_root_cause_mapped.json', 'w') as file:
#     json.dump(training_df.to_dict(orient='records'), file, indent=4)

# print("Processed training data with root causes saved to 'processed_logs_with_root_cause.json'.")



