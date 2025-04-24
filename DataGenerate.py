import json
import random

# Sample log messages and root causes
log_messages = [
    ("INFO exited: selenium-node (exit status 0; expected)", "Normal termination"),
    ("WARN received SIGINT indicating exit request", "Manual interruption"),
    ("CRIT Server 'unix_http_server' running without any HTTP authentication checking", "Authentication failure"),
    ("INFO [NodeServer.lambda$createHandlers$3] - Shutting down", "Node shutdown initiated"),
    ("WARN received SIGTERM indicating exit request", "Pod termination"),
    ("ERROR Connection to database lost, attempting to reconnect...", "Database connection issue"),
    ("INFO stopped: vnc (terminated by SIGTERM)", "VNC service shutdown"),
    ("CRIT Memory limit exceeded, process killed", "Out of memory (OOM)"),
    ("ERROR Failed to fetch data from API - timeout after 30s", "API timeout"),
    ("WARN Disk space usage exceeded 90%", "Disk space issue"),
    ("ERROR Unable to resolve DNS for service 'selenium-hub'", "DNS resolution failure"),
    ("CRIT Process 'webdriver' exited unexpectedly", "WebDriver crash"),
    ("WARN High CPU usage detected (95%)", "CPU overload"),
    ("ERROR Network unreachable - check internet connection", "Network failure"),
    ("INFO Selenium Grid started successfully on port 4444", "Successful startup"),
    ("WARN Unresponsive script detected, restarting process", "Hung process"),
    ("ERROR Authentication failed for user 'admin'", "Invalid credentials"),
    ("CRIT Kubernetes pod 'selenium-node' restarted due to crash", "Pod crash loop"),
    ("INFO Download complete: 'test-results.zip'", "File transfer successful"),
    ("ERROR SSL certificate verification failed", "Certificate validation issue"),
    ("WARN Session timeout exceeded, terminating connection", "Session expiration"),
    ("INFO User 'qa-engineer' logged in successfully", "Successful authentication"),
    ("CRIT Kernel panic - system rebooting", "Kernel failure"),
    ("ERROR WebSocket connection to 'browserstack' lost", "WebSocket disconnection"),
    ("INFO New Selenium session started: Session ID 12345", "Session creation"),
    ("WARN Low available memory: 512MB remaining", "Memory pressure"),
    ("INFO Uploaded file 'appium-test.apk' to S3 successfully", "Successful upload"),
    ("CRIT Docker container 'chrome-node' restarted due to failure", "Container crash"),
    ("ERROR Unable to allocate IP address from DHCP", "DHCP failure"),
    ("INFO Kubernetes deployment 'selenium-grid' updated successfully", "Deployment update"),
    ("WARN Browser instance unresponsive, force quitting", "Browser hang"),
    ("INFO Test execution completed: 50 passed, 2 failed", "Test run summary")
]

# Generate 1000 random log entries
log_data = [{"message": random.choice(log_messages)[0], "root_cause": random.choice(log_messages)[1]} for _ in range(1000)]

# Save to a JSON file
file_path = "/mnt/data/log_data.json"
with open(file_path, "w") as f:
    json.dump(log_data, f, indent=4)

file_path
