import time
import os
import numpy as np
from sanitization_logic import sanitization_logic

def run_sanitization_daemon():
    print("\n --Sanitization Service-- Daemon started. Running every 60 seconds...")
    
    output_file = "models/features/adaptive_batch.npz"
    os.makedirs("models/features", exist_ok=True)

    try:
        while True:
            # wait for the next cycle
            time.sleep(60) #have to change in real implementation to trigger based on new data in the database rather than fixed sleep
            
            print("\n --Sanitization Service-- Waking up to query the database for new events...")

            # The logic now communicates directly with the database (without arguments)
            X_clean, y_clean = sanitization_logic()

            # We save the filtered data for the adaptive_learning process
            if X_clean is not None and len(X_clean) > 0:
                np.savez_compressed(output_file, X=X_clean, y=y_clean)
                print(f"\n [Sanitization Service] Success: {len(X_clean)} new samples saved in {output_file}.")
            else:
                print("\n [Sanitization Service] Skipped: No confident data available in the database this cycle.")

    except KeyboardInterrupt:
        print("\n --Sanitization Service-- Shutting down... \n")

if __name__ == "__main__":
    run_sanitization_daemon()