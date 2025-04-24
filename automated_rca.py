import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib

# Load the trained model and vectorizer (assuming LDA with CountVectorizer)
vectorizer = joblib.load("count_vectorizer.pkl")  # Pre-trained CountVectorizer
lda_model = joblib.load("lda_model_tuned.pkl")  # Pre-trained LDA model

# Load the test data file
with open('pre_process_test_data.json', 'r') as file:
    test_data = json.load(file)

# Convert the test data into a DataFrame for processing
test_df = pd.DataFrame(test_data)

# Verify if 'message' column exists in the test data
if 'message' not in test_df.columns:
    raise ValueError("The column 'message' does not exist in the test data.")

# Transform the 'message' text using the vectorizer
X_test_features = vectorizer.transform(test_df['message'])

# Predict topics (or root causes) using the trained model
test_df['root_cause'] = lda_model.transform(X_test_features).argmax(axis=1)

#Define initial topic-to-root-cause mapping
topic_to_root_cause = {
      0: "SSL certificate verification failed",
      1: "Unable to allocate IP address from DHCP",
      2: "Connection to database lost, attempting to reconnect...",
      3: "Kubernetes pod 'selenium-hub' in CrashLoopBackOff",
      4: "DNS lookup delay detected",
      5: "Database migration failed",
      6: "received SIGTERM indicating exit request",
      7: "ebSocket connection to 'browserstack' lost",
      8: "Network unreachable - check internet connection",
      9: "Low available memory: 512MB remaining",
      10: "Unable to connect to Appium server",
      11: "Authentication failed for user 'admin'",
      12: "Appium driver initialization failed",
      13: "Docker container restart limit reached",
      14: "Proxy server connection unstable",
      15: "Browser instance unresponsive, force quitting",
      16: "Unresponsive script detected, restarting process",
      17: "Failed to fetch data from API - timeout after 30s",
      18: "Configuration file 'grid-config.yml' missing required fields",
      19: "Disk space usage exceeded 90%",
      20: "Browser process exceeded execution limit, terminating",
      21: "Certificate expired, TLS connection rejected",
      22: "Failed to upload log file to cloud storage",
      23: "Selenium session timeout reached, terminating test",
      24: "Unable to resolve DNS for service 'selenium-hub'",
      25: "Test execution halted due to missing test data",
      26: "Network bandwidth exceeded limit",
      27: "High CPU usage detected (95%)",
      28: "Session timeout exceeded, terminating connection",
      29: "received SIGINT indicating exit request",
      30: "High memory usage detected: 95%"
}

# Map predicted topics to root causes
test_df['root_cause'] = test_df['root_cause'].map(topic_to_root_cause)

# Assign 'nan' to rows where no root cause mapping was found
test_df['root_cause'] = test_df['root_cause'].fillna('nan')

# Convert the DataFrame back to JSON format
updated_test_data = test_df.to_dict(orient='records')

# Save the updated test data with root causes back to the JSON file
with open('test_data_with_root_cause.json', 'w') as file:
    json.dump(updated_test_data, file, indent=4)

print("Updated test data saved to 'test_data_with_root_cause.json'.")

# Read the JSON file and extract the "root_cause" values
with open('test_data_with_root_cause.json', 'r') as file:
    test_data_with_root_cause = json.load(file)

# Extract unique root causes from the file
extracted_root_causes = {entry["root_cause"] for entry in test_data_with_root_cause if "root_cause" in entry}

# Merge extracted root causes into topic_to_root_cause mapping
for idx, root_cause in enumerate(extracted_root_causes, start=len(topic_to_root_cause)):
    if root_cause not in topic_to_root_cause.values():
        topic_to_root_cause[idx] = root_cause

# Print the updated mapping
print("Updated topic_to_root_cause mapping:", topic_to_root_cause)
