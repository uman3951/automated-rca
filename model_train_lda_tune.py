import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import GridSearchCV
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

# Set up the LDA model
lda_model = LatentDirichletAllocation(random_state=42)

# Define the hyperparameter grid for tuning
param_grid = {
    'n_components': [10, 30, 50],  # Number of topics
    'learning_method': ['online'],  # Learning methods
    'learning_offset': [20.0, 100.0],  # Control the speed of learning
    'max_iter': [50, 100, 200],  # Number of iterations
    'doc_topic_prior': [0.01, 0.1, 0.2],  # Alpha (topic priors)
    'topic_word_prior': [0.01, 0.1, 0.2]  # Beta (word priors)
}

# param_grid = {
#     'n_components': [10, 20],  
#     'learning_method': ['online'],  
#     'max_iter': [50, 100],  
#     'doc_topic_prior': [0.01, 0.5],  
#     'topic_word_prior': [0.01, 0.5]  
# }

# Set up the GridSearchCV with cross-validation
grid_search = GridSearchCV(lda_model, param_grid, cv=3, n_jobs=2, verbose=1)

# Fit the GridSearchCV to the features
grid_search.fit(X_features)

# Check if the grid search has completed and output the best parameters
print("Best parameters found by GridSearchCV:")
print(grid_search.best_params_)

# Ensure best_estimator is found
best_lda_model = grid_search.best_estimator_

# Save the vectorizer and the best tuned LDA model
if best_lda_model:
    joblib.dump(vectorizer, "count_vectorizer.pkl")
    joblib.dump(best_lda_model, "lda_model_tuned.pkl")
    print("Models saved successfully as 'count_vectorizer.pkl' and 'lda_model_tuned.pkl'.")
else:
    print("Error: Best model not found. Please check grid search results.")

# Assign topics (or root causes) to the training data using the best tuned model
topic_probabilities = best_lda_model.transform(X_features)
training_df['topic_index'] = topic_probabilities.argmax(axis=1)


training_df['root_cause'] = training_df['topic_index'].map(root_cause_mapping)

# Optional: Drop the topic index column if not needed
# training_df.drop(columns=['topic_index'], inplace=True)

# Save the training data with root causes back to a file
with open('pre_processed_logs_with_root_cause_mapped_tuned.json', 'w') as file:
    json.dump(training_df.to_dict(orient='records'), file, indent=4)

print("Processed training data with root causes saved to 'pre_processed_logs_with_root_cause_mapped_tuned.json'.")

