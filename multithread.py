import threading
import queue
from collections import deque
import pandas as pd
import numpy as np
import time
from live_sniffer import capture_live_traffic
from libpcap_approach import libpcap_capture
import non_linear as nl
import time 

# ---------------------------------------------------------------------------------------------------------

#Thread-Safe Queue for transferring DataFrames from Sniffer to Analyzer
data_queue = queue.Queue()

# 2. Producer Thread: Live traffic capture
def sniffer_worker(duration, interface):
    print(f"\n Active on {interface} | Batch duration: {duration}s")
    while True:
        # Capture and extract flow features
        df_batch = libpcap_capture(capture_duration=duration, interface=interface)
        
        if not df_batch.empty:
            # Place the batch in the queue for processing
            data_queue.put(df_batch)


# ---------------------------------------------------------------------------------------------------------


# Consumer Thread: Analysis & Detection
# Optimized to receive pre-loaded model, scaler, and training metrics
def analyzer_worker(w_train, sigma_train, mean_train, raw_mean, raw_std, theta_cheb, model, scaler):
    # Sliding window buffer (traffic history)
    buffer = deque(maxlen=150) 
    print("\n Linear Regression Model initialized \n")
    
    while True:
        # Get new batch of data from the sniffer
        new_df = data_queue.get()
        
        # Update the buffer (past + present)
        for _, row in new_df.iterrows():
            buffer.append(row.values)
        
        # Start analysis if buffer has the minimum window size
        if len(buffer) >= w_train:
            current_data = np.array(list(buffer))
            
            # compute dynamic window size based on live volatility
            sigma_live = np.std(current_data, axis=0).mean()
            w_dynamic = int(w_train * (sigma_live / sigma_train.mean()))
            
            # w_dynamic does not exceed current buffer size or limits
            w_dynamic = max(15, min(w_dynamic, 50, len(buffer))) 
            
            # select the current analysis window
            analysis_batch = current_data[-w_dynamic:]
            actual_w = len(analysis_batch) # Get size of the slice
            
            T_vol = analysis_batch[:, 0]
            N_req = analysis_batch[:, 1]
            S_len = analysis_batch[:, 2]

            # --- First Stability Check (Residuals vs Threshold) ---
            
            current_window_mean = np.mean(analysis_batch,axis=0)
            res_diff = np.abs(current_window_mean - raw_mean)

            
            # Calculate first stage residuals
            threshold1_f = 0.1 * raw_std

            print(f"\n[DEBUG] Raw Mean (Trained): {raw_mean}")
            print(f"[DEBUG] Current Mean (Live):  {current_window_mean}")
            print(f"[DEBUG] Difference (Res_diff):{res_diff}")
            print(f"[DEBUG] Threshold:          {threshold1_f}")
            print("-" * 40)
            
           # check to trigger second stage
            if np.any(res_diff > threshold1_f):
                print(f"\n Stage 1: Unstable traffic detected \n")
                
                start_time = time.time()
                
                anomaly_type = nl.signature_analysis(analysis_batch, raw_mean, raw_std)
                
                if "Unknown" not in anomaly_type:
                    end_time = time.time() - start_time
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    log_message = f"[{timestamp}] ALERT: {anomaly_type} | Detected via Volume Signature\n"
                    with open("ids_alerts.log", "a", encoding="utf-8") as log_file:
                        log_file.write(log_message)
                        
                    print(f"!!! Anomaly Detected | Type: {anomaly_type}")
                
                else:
                    entropy_res = nl.entropy_based_stab_check(T_vol, N_req, S_len, actual_w, model, scaler)
                    
                    # compute Z-score
                    z_score = (entropy_res - mean_train) / sigma_train
                    
                    if np.any(z_score > theta_cheb):
                        end_time = time.time() - start_time
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        max_z = np.max(z_score)
                        
                        log_message = f"[{timestamp}] ALERT: Stealth Entropy Anomaly | Max Z-Score: {max_z:.2f} | Threshold: {theta_cheb:.2f}\n"
                        with open("ids_alerts.log", "a", encoding="utf-8") as log_file:
                            log_file.write(log_message)
                            
                        print(f"!!! Anomaly Detected | Type: Stealth Entropy Shift (Z: {max_z:.2f})")
                    else:
                        end_time = time.time() - start_time
                        print(f" Stage 2: Everything is normal | Time: {end_time:.8f} seconds \n")

            else:
                print(f" Stage 1: Traffic is stable \n")
        
        # signal that the batch processing is complete
        data_queue.task_done()


# ---------------------------------------------------------------------------------------------------------


# System Start Execution
def start_live_ids():
    # load training resources at startup to optimize performance
    print("\n Loading model, scaler, and training metrics...")
    model, scaler = nl.load_model_and_scaler()
    
    # load metrics from training with feature expansion
    try:
        params = np.load('train_metrics_feature_expansion.npz')
        sigma_train = params['sigma_train']
        mean_train = params['mean_train']
        raw_mean = params['raw_mean']
        raw_std = params['raw_std']
    except Exception as e:
        print(e)
        return 
    
    # compute Chebyshev threshold
    theta_cheb = nl.theta_cheb(0.05)
    
    # training window size
    w_train = 20 

    # Thread 1: Sniffer 
    t_sniff = threading.Thread(target=sniffer_worker, args=(2, 'lo0'), daemon=True)

    # Thread 2: Analyzer 
    t_analyze = threading.Thread(
        target=analyzer_worker, 
        args=(w_train, sigma_train, mean_train, raw_mean, raw_std, theta_cheb, model, scaler), 
        daemon=True
    )

    t_sniff.start()
    t_analyze.start()

    print("\n IDS is live and running \n")

    try:
        while True: 
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n Stopping Live IDS... \n")


# ---------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    start_live_ids()