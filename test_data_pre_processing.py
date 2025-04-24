import json
import re
from datetime import datetime


# Process the training data
def process_logs(data):
    """Parse and process logs."""
    processed_logs = []

    for hit in data.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        log_level_match = re.search(r"(INFO|CRIT|ERROR|WARN)", source.get("log", ""))
        log_level = log_level_match.group(0) if log_level_match else "UNKNOWN"

        log_entry = {
            "timestamp": source.get("time"),
            "log_level": log_level,
            "message": source.get("log", ""),
            "root_cause": "",
            "container_name": source.get("kubernetes", {}).get("container_name"),
            "namespace": source.get("kubernetes", {}).get("namespace_name"),
            "pod_name": source.get("kubernetes", {}).get("pod_name"),
        }
        processed_logs.append(log_entry)

    return processed_logs

# Save the processed logs to a file
def save_logs(processed_logs, output_file):
    """Save processed logs to a file."""
    with open(output_file, "w") as f:
        json.dump(processed_logs, f, indent=4)

# Main execution
if __name__ == "__main__":
    # Load training data
    with open("sample_log.json", "r") as f:
        training_data = json.load(f)

    # Process the logs
    processed_logs = process_logs(training_data)

    # Save the processed logs
    save_logs(processed_logs, "processed_test_logs.json")
    print(f"Processed logs saved to 'processed_test_logs.json'")
