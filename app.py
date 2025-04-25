from flask import Flask, jsonify, request
import json
import pandas as pd
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib

app = Flask(__name__)

# # Load trained model and vectorizer
# vectorizer = joblib.load("count_vectorizer.pkl")
# #lda_model = joblib.load("lda_model.pkl")
# lda_tuned_model = joblib.load("lda_model_tuned.pkl")  # Load the tuned LDA model

# INPUT_DIR = "logs"  # Directory containing log files
# PROCESSED_TRAINING_DATA = "processed_traning_data.json"

# # Paths for saved models
# VECTORIZER_PATH = "count_vectorizer.pkl"
# # LDA_MODEL_PATH = "lda_model.pkl"
# LDA_TUNED_MODEL_PATH = 'lda_model_tuned.pkl'

# #Training Data set Path
# TRAINING_DATA_PATH = "processed_traning_data.json"

# # Topic-to-root cause mapping
# topic_to_root_cause = {}

# # Read the JSON file and extract the "root_cause" values
# with open('processed_traning_data.json', 'r') as file:
#     test_data_with_root_cause = json.load(file)

# # Extract unique root causes from the file
# extracted_root_causes = {entry["root_cause"] for entry in test_data_with_root_cause if "root_cause" in entry}

# # Merge extracted root causes into topic_to_root_cause mapping
# for idx, root_cause in enumerate(extracted_root_causes, start=len(topic_to_root_cause)):
#     if root_cause not in topic_to_root_cause.values() :
#         topic_to_root_cause[idx] = root_cause

@app.route('/')
def home():
    return "Welcome to the Automated RCA Prediction API. Use /predict or /predict_tuned to get root causes.", 200

@app.route('/favicon.ico')
def favicon():
    return '', 204

# Load model and vectorizer once
vectorizer = joblib.load("count_vectorizer.pkl")
lda_model = joblib.load("lda_model_tuned.pkl")

def load_topic_mapping_from_s3_public(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch JSON from {url}, status code: {response.status_code}")
    topic_map = json.loads(response.text)
    return {int(k): v for k, v in topic_map.items()}

def upload_file_via_presigned_url(local_file_path, presigned_url):
    with open(local_file_path, 'rb') as file_data:
        response = requests.put(presigned_url, data=file_data)
    if response.status_code == 200:
        print("File uploaded successfully.")
        return True
    else:
        print(f"Upload failed: {response.status_code}, {response.text}")
        return False

@app.route("/run_rca", methods=["POST"])
def run_rca():
    # Load test data
    # try:
    #     with open('pre_process_test_data.json', 'r') as file:
    #         test_data = json.load(file)
    # except Exception as e:
    #     return jsonify({"error": f"Failed to read test data: {str(e)}"}), 500
    
    try:
        s3_url = "https://udaraquest1.s3.us-east-1.amazonaws.com/pre_process_test_data.json"
        response = requests.get(s3_url)
        if response.status_code != 200:
            return jsonify({"error": f"Failed to fetch test data from S3: {response.status_code}"}), 500
        test_data = response.json()
    except Exception as e:
        return jsonify({"error": f"Failed to read test data from S3: {str(e)}"}), 500

    test_df = pd.DataFrame(test_data)
    if 'message' not in test_df.columns:
        return jsonify({"error": "'message' column missing in input"}), 400

    # Predict root cause
    X_test_features = vectorizer.transform(test_df['message'])
    test_df['root_cause'] = lda_model.transform(X_test_features).argmax(axis=1)

    # Load topic mapping from S3
    try:
        topic_to_root_cause = load_topic_mapping_from_s3_public(
            "https://udaraquest1.s3.amazonaws.com/topic_to_root_cause.json"
        )
    except Exception as e:
        return jsonify({"error": f"Failed to load topic mapping: {str(e)}"}), 500

    test_df['root_cause'] = test_df['root_cause'].map(topic_to_root_cause).fillna('nan')

    # Save locally
    local_output_file = 'test_data_with_root_cause.json'
    with open(local_output_file, 'w') as file:
        json.dump(test_df.to_dict(orient='records'), file, indent=4)

    # Upload using provided presigned URL
    presigned_url = request.json.get("presigned_url")
    if not presigned_url:
        return jsonify({"error": "Missing presigned_url in request body"}), 400

    upload_success = upload_file_via_presigned_url(local_output_file, presigned_url)
    if not upload_success:
        return jsonify({"error": "File upload failed"}), 500

    return jsonify({
        "message": "Root cause prediction complete and file uploaded",
        "root_causes": test_df['root_cause'].unique().tolist()
    })



# @app.route('/predict_lda', methods=['POST'])
# def predict_root_cause():
#     try:
#         input_data = request.get_json()
#         if not input_data:
#             return jsonify({"error": "Empty input data"}), 400

#         test_df = pd.DataFrame(input_data)
#         if 'message' not in test_df.columns:
#             return jsonify({"error": "Missing 'message' column in input data"}), 400

#         # LDA Model Prediction
#         X_test_features = vectorizer.transform(test_df['message'])

#         # Debugging: Print the shape and vocabulary size
#         print(f"Shape of transformed features: {X_test_features.shape}")
#         print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")  
        
#         # Remove the 26 features check to allow dynamic feature count
#         test_df['lda_root_cause'] = lda_model.transform(X_test_features).argmax(axis=1)
#         test_df['lda_root_cause'] = test_df['lda_root_cause'].map(topic_to_root_cause).fillna('Unknown')

#         # Save the result to "test_data_final_result_lda.json"
#         with open('test_data_final_result_lda.json', 'w') as outfile:
#             json.dump(test_df.to_dict(orient='records'), outfile, indent=4)

#         return jsonify(test_df.to_dict(orient='records'))

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # NEW ENDPOINT: Predict Root Cause Using Tuned LDA Model
# @app.route('/predict_lda_tuned', methods=['POST'])
# def predict_tuned_root_cause():
#     try:
#         input_data = request.get_json()
#         if not input_data:
#             return jsonify({"error": "Empty input data"}), 400

#         test_df = pd.DataFrame(input_data)
#         if 'message' not in test_df.columns:
#             return jsonify({"error": "Missing 'message' column in input data"}), 400
        
#         # Fit on training data
#         vectorizer.fit(train_df['message'])

#         # Tuned LDA Model Prediction
#         X_test_features = vectorizer.transform(test_df['message'])
        
#         # Remove the 26 features check to allow dynamic feature count
#         test_df['lda_tuned_root_cause'] = lda_tuned_model.transform(X_test_features).argmax(axis=1)
#         test_df['lda_tuned_root_cause'] = test_df['lda_tuned_root_cause'].map(topic_to_root_cause).fillna('Unknown')

#         # Save the result to "test_data_final_lda_tuned.json"
#         with open('test_data_final_lda_tuned.json', 'w') as outfile:
#             json.dump(test_df.to_dict(orient='records'), outfile, indent=4)

#         return jsonify(test_df.to_dict(orient='records'))
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
    
# # ENDPOINT: Evaluate Perplexity for LDA
# @app.route('/evaluate_perplexity_lda', methods=['POST'])
# def evaluate_perplexity_lda():
#     try:
#         input_data = request.get_json()
#         if not input_data:
#             return jsonify({"error": "Empty input data"}), 400

#         test_df = pd.DataFrame(input_data)
#         if 'message' not in test_df.columns:
#             return jsonify({"error": "Missing 'message' column in input data"}), 400

#         X_test_features = vectorizer.transform(test_df['message'])
#         perplexity = lda_model.perplexity(X_test_features)

#         return jsonify({"perplexity": perplexity})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# # ENDPOINT: Evaluate Perplexity for LDA TUNED 
# @app.route('/evaluate_perplexity_lda_tuned', methods=['POST'])
# def evaluate_perplexity_lda_tuned():
#     try:
#         input_data = request.get_json()
#         if not input_data:
#             return jsonify({"error": "Empty input data"}), 400

#         test_df = pd.DataFrame(input_data)
#         if 'message' not in test_df.columns:
#             return jsonify({"error": "Missing 'message' column in input data"}), 400

#         X_test_features = vectorizer.transform(test_df['message'])
#         perplexity = lda_tuned_model.perplexity(X_test_features)

#         return jsonify({"perplexity": perplexity})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # ENDPOINT: Evaluate Log-Likelihood for LDA
# @app.route('/evaluate_log_likelihood_lda', methods=['POST'])
# def evaluate_log_likelihood_lda():
#     try:
#         input_data = request.get_json()
#         if not input_data:
#             return jsonify({"error": "Empty input data"}), 400

#         test_df = pd.DataFrame(input_data)
#         if 'message' not in test_df.columns:
#             return jsonify({"error": "Missing 'message' column in input data"}), 400

#         X_test_features = vectorizer.transform(test_df['message'])
#         log_likelihood = lda_model.score(X_test_features)

#         return jsonify({"log_likelihood": log_likelihood})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# # ENDPOINT: Evaluate Log-Likelihood for LDA Tuned
# @app.route('/evaluate_log_likelihood_lda_tuned', methods=['POST'])
# def evaluate_log_likelihood_lda_tuned():
#     try:
#         input_data = request.get_json()
#         if not input_data:
#             return jsonify({"error": "Empty input data"}), 400

#         test_df = pd.DataFrame(input_data)
#         if 'message' not in test_df.columns:
#             return jsonify({"error": "Missing 'message' column in input data"}), 400

#         X_test_features = vectorizer.transform(test_df['message'])
#         log_likelihood = lda_tuned_model.score(X_test_features)

#         return jsonify({"log_likelihood": log_likelihood})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # Get trained vocabulary
# trained_vocab = set(vectorizer.get_feature_names_out())

# @app.route('/evaluate_log_likelihood_lda_tuned', methods=['POST'])
# def evaluate_log_likelihood_lda_tuned():
#     try:
#         input_data = request.get_json()
#         if not input_data:
#             return jsonify({"error": "Empty input data"}), 400

#         test_df = pd.DataFrame(input_data)
#         if 'message' not in test_df.columns:
#             return jsonify({"error": "Missing 'message' column in input data"}), 400

#         # Keep only words present in the trained vocabulary
#         test_df['message'] = test_df['message'].apply(lambda msg: ' '.join(
#             [word for word in msg.split() if word in trained_vocab]
#         ))

#         # Transform the cleaned test data
#         X_test_features = vectorizer.transform(test_df['message'])

#         # Check feature size before scoring
#         if X_test_features.shape[1] != lda_tuned_model.n_features_in_:
#             return jsonify({
#                 "error": f"Feature size mismatch: Expected {lda_tuned_model.n_features_in_}, got {X_test_features.shape[1]}"
#             }), 400

#         log_likelihood = lda_tuned_model.score(X_test_features)

#         return jsonify({"log_likelihood": log_likelihood})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # ENDPOINT: Plot Perplexity LDA
# @app.route('/plot_perplexity_lda', methods=['GET'])
# def plot_perplexity_lda():
#     try:
#         topic_range = range(2, 40)
#         perplexities = []

#         # Convert test_data_with_root_cause to a DataFrame
#         test_df = pd.DataFrame(test_data_with_root_cause)

#         if 'message' not in test_df.columns:
#             return jsonify({"error": "Missing 'message' column in test data"}), 400

#         X_test_features = vectorizer.transform(test_df['message'])

#         for num_topics in topic_range:
#             temp_lda = joblib.load("lda_model.pkl")  # Reload model
#             perplexities.append(temp_lda.perplexity(X_test_features))

#         plt.figure(figsize=(8, 5))
#         plt.plot(topic_range, perplexities, marker='o', linestyle='-')
#         plt.xlabel("Number of Topics")
#         plt.ylabel("Perplexity")
#         plt.title("LDA Model Perplexity")
#         plt.grid()

#         plot_path = "perplexity_plot.png"
#         plt.savefig(plot_path)
#         plt.close()

#         return send_file(plot_path, mimetype='image/png')

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# # ENDPOINT: Plot Perplexity LDA tuned
# @app.route('/plot_perplexity_lda_tuned', methods=['GET'])
# def plot_perplexity_lda_tuned():
#     try:
#         topic_range = range(2, 15)
#         perplexities = []

#         # Convert test_data_with_root_cause to a DataFrame
#         test_df = pd.DataFrame(test_data_with_root_cause)

#         if 'message' not in test_df.columns:
#             return jsonify({"error": "Missing 'message' column in test data"}), 400

#         X_test_features = vectorizer.transform(test_df['message'])

#         for num_topics in topic_range:
#             temp_lda = joblib.load("lda_model.pkl")  # Reload model
#             perplexities.append(temp_lda.perplexity(X_test_features))

#         plt.figure(figsize=(8, 5))
#         plt.plot(topic_range, perplexities, marker='o', linestyle='-')
#         plt.xlabel("Number of Topics")
#         plt.ylabel("Perplexity")
#         plt.title("LDA Model Perplexity")
#         plt.grid()

#         plot_path = "perplexity_plot.png"
#         plt.savefig(plot_path)
#         plt.close()

#         return send_file(plot_path, mimetype='image/png')

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# # ENDPOINT: Plot Log-Likelihood LDA
# @app.route('/plot_log_likelihood_lda', methods=['GET'])
# def plot_log_likelihood():
#     try:
#         topic_range = range(2, 15)
#         log_likelihoods = []

#         # Convert test_data_with_root_cause to a DataFrame
#         test_df = pd.DataFrame(test_data_with_root_cause)

#         if 'message' not in test_df.columns:
#             return jsonify({"error": "Missing 'message' column in test data"}), 400

#         X_test_features = vectorizer.transform(test_df['message'])

#         for num_topics in topic_range:
#             temp_lda = joblib.load("lda_model.pkl")  # Reload model
#             log_likelihoods.append(temp_lda.score(X_test_features))

#         plt.figure(figsize=(8, 5))
#         plt.plot(topic_range, log_likelihoods, marker='o', linestyle='-')
#         plt.xlabel("Number of Topics")
#         plt.ylabel("Log-Likelihood")
#         plt.title("LDA Model Log-Likelihood")
#         plt.grid()

#         plot_path = "log_likelihood_plot.png"
#         plt.savefig(plot_path)
#         plt.close()

#         return send_file(plot_path, mimetype='image/png')

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# # ENDPOINT: Plot Log-Likelihood LDA Tuned
# @app.route('/plot_log_likelihood_lda_tuned', methods=['GET'])
# def plot_log_likelihood_lda_tuned():
#     try:
#         topic_range = range(2, 15)
#         log_likelihoods = []

#         # Convert test_data_with_root_cause to a DataFrame
#         test_df = pd.DataFrame(test_data_with_root_cause)

#         if 'message' not in test_df.columns:
#             return jsonify({"error": "Missing 'message' column in test data"}), 400

#         X_test_features = vectorizer.transform(test_df['message'])

#         for num_topics in topic_range:
#             temp_lda = joblib.load("lda_model.pkl")  # Reload model
#             log_likelihoods.append(temp_lda.score(X_test_features))

#         plt.figure(figsize=(8, 5))
#         plt.plot(topic_range, log_likelihoods, marker='o', linestyle='-')
#         plt.xlabel("Number of Topics")
#         plt.ylabel("Log-Likelihood")
#         plt.title("LDA Model Log-Likelihood")
#         plt.grid()

#         plot_path = "log_likelihood_plot.png"
#         plt.savefig(plot_path)
#         plt.close()

#         return send_file(plot_path, mimetype='image/png')

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


# # Endpoint to evaluate Perplexity for a specified model
# @app.route('/evaluate_perplexity', methods=['POST'])
# def evaluate_perplexity():
#     try:
#         input_data = request.get_json()
#         if not input_data:
#             return jsonify({"error": "Empty input data"}), 400

#         # Extract model type from the query parameter
#         model_type = request.args.get('model', default='lda', type=str)
#         if model_type == 'lda_tuned':
#             model = lda_tuned_model
#         else:
#             model = lda_model

#         test_df = pd.DataFrame(input_data)
#         if 'message' not in test_df.columns:
#             return jsonify({"error": "Missing 'message' column in input data"}), 400

#         X_test_features = vectorizer.transform(test_df['message'])
#         perplexity = model.perplexity(X_test_features)

#         return jsonify({"perplexity": perplexity})

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
    
# ###################### Training Data Processing #############################



# # Function to process logs
# def process_logs(data):
#     """Parse and process logs."""
#     processed_logs = []

#     for hit in data.get("hits", {}).get("hits", []):
#         source = hit.get("_source", {})
#         log_level_match = re.search(r"(CRIT|ERROR|WARN)", source.get("log", ""))
#         log_level = log_level_match.group(0) if log_level_match else "UNKNOWN"

#         log_entry = {
#             "timestamp": source.get("time"),
#             "log_level": log_level,
#             "message": source.get("log", ""),
#             "root_cause": "",
#             "container_name": source.get("kubernetes", {}).get("container_name"),
#             "namespace": source.get("kubernetes", {}).get("namespace_name"),
#             "pod_name": source.get("kubernetes", {}).get("pod_name"),
#         }
#         processed_logs.append(log_entry)

#     return processed_logs

# # Function to save processed logs
# def save_process_logs(processed_logs):
#     """Append processed logs to the output file."""
#     existing_logs = []

#     if os.path.exists(PROCESSED_TRAINING_DATA):
#         with open(PROCESSED_TRAINING_DATA, "r") as f:
#             try:
#                 existing_logs = json.load(f)
#             except json.JSONDecodeError:
#                 existing_logs = []

#     existing_logs.extend(processed_logs)

#     with open(PROCESSED_TRAINING_DATA, "w") as f:
#         json.dump(existing_logs, f, indent=4)

# # Flask API endpoint to process logs
# @app.route('/process_training_logs', methods=['POST'])
# def process_logs_api():
#     """Process all log files in the directory and save results."""
#     all_processed_logs = []

#     for file_name in os.listdir(INPUT_DIR):
#         if file_name.endswith(".json"):
#             file_path = os.path.join(INPUT_DIR, file_name)
#             with open(file_path, "r") as f:
#                 try:
#                     training_data = json.load(f)
#                     all_processed_logs.extend(process_logs(training_data))
#                 except json.JSONDecodeError:
#                     return jsonify({"error": f"Invalid JSON in file {file_name}"}), 400

#     if all_processed_logs:
#         save_process_logs(all_processed_logs)
#         return jsonify({"message": "Logs processed successfully", "total_logs": len(all_processed_logs)}), 200
#     else:
#         return jsonify({"message": "No valid logs found to process"}), 400

# # Flask API endpoint to retrieve processed logs
# @app.route('/get_logs', methods=['GET'])
# def get_logs():
#     """Retrieve the processed logs."""
#     if os.path.exists(PROCESSED_TRAINING_DATA):
#         with open(PROCESSED_TRAINING_DATA, "r") as f:
#             try:
#                 logs = json.load(f)
#                 return jsonify(logs), 200
#             except json.JSONDecodeError:
#                 return jsonify({"error": "Processed logs file is corrupted"}), 500
#     return jsonify({"message": "No processed logs found"}), 404

############################# Update Root Cause ##################################

# # Flask API endpoint to update root cause for matching log messages (partial match)
# @app.route('/update_root_cause', methods=['POST'])
# def update_root_cause():
#     """Update root cause in all logs matching a partial message."""
#     if not os.path.exists(PROCESSED_TRAINING_DATA):
#         return jsonify({"error": "No processed logs found"}), 404

#     data = request.get_json()
#     if not data or "message" not in data or "root_cause" not in data:
#         return jsonify({"error": "Invalid request, 'message' and 'root_cause' are required"}), 400

#     message_to_update = data["message"]
#     new_root_cause = data["root_cause"]

#     try:
#         with open(PROCESSED_TRAINING_DATA, "r") as f:
#             logs = json.load(f)

#         updated_logs = []
#         count = 0
#         for log in logs:
#             if message_to_update in log.get("message", ""):  # Partial match (substring)
#                 log["root_cause"] = new_root_cause
#                 count += 1
#             updated_logs.append(log)

#         # Save the updated logs back to the file
#         save_logs(updated_logs, PROCESSED_TRAINING_DATA)

#         if count == 0:
#             return jsonify({"message": "No logs found matching the given message"}), 404
#         return jsonify({"message": "Root cause updated successfully", "updated_logs": count}), 200

#     except json.JSONDecodeError:
#         return jsonify({"error": "Processed logs file is corrupted"}), 500

# # Function to save the logs to a file
# def save_logs(logs, output_file):
#     """Save the processed logs to a specified file."""
#     with open(output_file, "w") as f:
#         json.dump(logs, f, indent=4)
#     print(f"Processed logs saved to '{output_file}'")
#     print(f"Updated file content: {json.dumps(logs, indent=4)}")

# # Function to extract unique WARN and ERROR messages
# def get_distinct_messages():
#     unique_messages = set()

#     if os.path.exists(PROCESSED_TRAINING_DATA):
#         with open(PROCESSED_TRAINING_DATA, "r") as f:
#             try:
#                 logs = json.load(f)
#                 for log in logs:
#                     message = log.get("message", "")
                    
#                     # Extract message after "WARN" or "ERROR"
#                     match = re.search(r"(WARN|ERROR)\s+(.+)", message)
#                     if match:
#                         unique_messages.add(match.group(2))  # Get the extracted part

#             except json.JSONDecodeError:
#                 return {"error": "Processed logs file is corrupted"}, 500

#     return list(unique_messages)

# # Flask API endpoint to retrieve distinct WARN and ERROR messages
# @app.route('/get_distinct_messages', methods=['GET'])
# def get_messages():
#     """Retrieve distinct messages after WARN or ERROR."""
#     messages = get_distinct_messages()
    
#     if messages:
#         return jsonify({"distinct_messages": messages}), 200
#     return jsonify({"message": "No valid messages found"}), 404

# ######################### Model Training #############

# def train_model(model_type="lda"):
#     """Train the LDA model with training data."""
#     # Load training data
#     if not os.path.exists(TRAINING_DATA_PATH):
#         return "Training data file not found.", 400
    
#     with open(TRAINING_DATA_PATH, 'r') as file:
#         training_data = json.load(file)
    
#     training_df = pd.DataFrame(training_data)
    
#     if 'message' not in training_df.columns or 'root_cause' not in training_df.columns:
#         return "Missing required columns in training data.", 400
    
#     training_df['root_cause'] = training_df['root_cause'].fillna("Unknown Issue")
#     unique_root_causes = training_df['root_cause'].unique()
#     root_cause_mapping = {i: cause for i, cause in enumerate(unique_root_causes)}
    
#     vectorizer = CountVectorizer(max_features=1000, stop_words='english')
#     X_features = vectorizer.fit_transform(training_df['message'])
    
#     # Initialize and fit the model based on the model_type
#     lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
#     lda_model.fit(X_features)
    
#     # Save the vectorizer
#     joblib.dump(vectorizer, VECTORIZER_PATH)
    
#     # Save the appropriate LDA model based on the model_type
#     if model_type == 'lda_tuned':
#         joblib.dump(lda_model, LDA_TUNED_MODEL_PATH)
#     else:
#         joblib.dump(lda_model, LDA_MODEL_PATH)
    
#     return "Model training complete. Models saved.", 200

# @app.route('/model_train', methods=['POST'])
# def train():
#     # Get model_type from request parameters
#     model_type = request.args.get('model_type', default='lda', type=str)
    
#     if model_type not in ['lda', 'lda_tuned']:
#         return jsonify({"error": "Invalid model type. Choose 'lda' or 'lda_tuned'."}), 400
    
#     message, status = train_model(model_type)
#     return jsonify({"message": message}), status

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)