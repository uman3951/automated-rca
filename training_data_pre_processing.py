import json
import re
from datetime import datetime

def get_root_cause(log_message, log_level):
    if log_level in ["ERROR", "WARN"]:
        return "Possible root cause needs manual analysis"
    elif log_level == "INFO":
        return "N/A"
    return None

def process_logs(data):
    processed_logs = []

    for hit in data.get("hits", {}).get("hits", []):
        source = hit.get("_source", {})
        # If need INFO as well 
        # log_level_match = re.search(r"(INFO|CRIT|ERROR|WARN)", source.get("log", ""))
        log_level_match = re.search(r"(CRIT|ERROR|WARN)", source.get("log", ""))
        log_level = log_level_match.group(0) if log_level_match else "UNKNOWN"

        log_entry = {
            "timestamp": source.get("time"),
            "log_level": log_level,
            "message": source.get("log", ""),
            "root_cause": get_root_cause(source.get("log", ""), log_level),
            "container_name": source.get("kubernetes", {}).get("container_name"),
            "namespace": source.get("kubernetes", {}).get("namespace_name"),
            "pod_name": source.get("kubernetes", {}).get("pod_name"),
        }
        processed_logs.append(log_entry)

    return processed_logs

def save_logs(processed_logs, output_file):
    with open(output_file, "w") as f:
        json.dump(processed_logs, f, indent=4)

if __name__ == "__main__":
    with open("sample_log1.json", "r") as f:
        training_data = json.load(f)

    processed_logs = process_logs(training_data)

    save_logs(processed_logs, "processed_logs.json")
    print(f"Processed logs saved to 'processed_logs.json'")
