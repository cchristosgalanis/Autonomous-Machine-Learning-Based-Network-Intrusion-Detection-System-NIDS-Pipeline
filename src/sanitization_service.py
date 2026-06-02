import time
import os
import numpy as np
from sanitization_logic import sanitization_logic

def run_sanitization_daemon():
    print("\n [Sanitization Service] Daemon started. Running every 60 seconds...")
    
    master_log_path = "logs/master_traffic_records.jsonl"
    output_file = "models/features/adaptive_batch.npz"
    
    os.makedirs("logs", exist_ok=True)
    os.makedirs("features", exist_ok=True)

    try:
        while True:
            # wait 60sec
            time.sleep(60)
            
            print(f"\n [Sanitization Service] Waking up to process {master_log_path}...")
            
            if not os.path.exists(master_log_path):
                print("\n Master log not found yet. Skipping cycle.")
                continue

            # sanitize the current batch
            X_clean, y_clean = sanitization_logic(master_log_path)

            # empty the log file immediately after reading to prevent memory/disk bloat
            try:
                open(master_log_path, 'w').close()
                print(f" [+] Truncated master log: {master_log_path} (Disk space recovered)")
            except Exception as e:
                print(f" [!] Failed to truncate {master_log_path}: {e}")
            # ---------------------------

            #  save the sanitized batch for the adaptive_learning process
            if X_clean is not None and len(X_clean) > 0:
                np.savez_compressed(output_file, X=X_clean, y=y_clean)
                print(f" [SUCCESS] Sanitization complete. {len(X_clean)} samples ready in {output_file}.")
            else:
                print(" [i] Sanitization skipped or no confident data available in this cycle.")

    except KeyboardInterrupt:
        print("\n [Sanitization Service] Shutting down... \n")

if __name__ == "__main__":
    run_sanitization_daemon()