import threading
import queue
from collections import deque
import pandas as pd
import numpy as np
import time
from live_sniffer import capture_live_traffic
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
        df_batch = capture_live_traffic(capture_duration=duration, interface=interface)
        
        if not df_batch.empty:
            # Place the batch in the queue for processing
            data_queue.put(df_batch)


# ---------------------------------------------------------------------------------------------------------


# Consumer Thread: Analysis & Detection
# Optimized to receive pre-loaded model, scaler, and training metrics
def analyzer_worker(w_train, sigma_train, mean_train, theta_cheb, model, scaler):
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
            # calculate μ and σ using actual window size
            f_byts_m, f_byts_s = nl.norm_datas(T_vol, actual_w)
            f_pkts_m, f_pkts_s = nl.norm_datas(N_req, actual_w)
            f_dur_m, f_dur_s = nl.norm_datas(S_len, actual_w)

            norm_flows_mean = np.column_stack([f_byts_m, f_pkts_m, f_dur_m])
            norm_flows_std = np.column_stack([f_byts_s, f_pkts_s, f_dur_s])
            
            live_traffic_reshaped = analysis_batch.reshape(1, actual_w, 3)
            
            # Calculate first stage residuals
            residual_1 = nl.first_stab_check(norm_flows_mean, live_traffic_reshaped, actual_w)
            threshold1_f = 0.1 * norm_flows_std
            
            # check to trigger second stage
            if np.any(residual_1 > threshold1_f[:, np.newaxis, :]):
                print(f"\n Stage 1: Unstable traffic detected \n")
                
                # --- Entropy-Based Regression Check ---
                start_time = time.time()
                entropy_res = nl.entropy_based_stab_check(T_vol, N_req, S_len, actual_w, model, scaler)
                
                # compute Z-score calculation using pre-loaded metrics
                z_score = (entropy_res - mean_train) / sigma_train

                # check for anomalies using Chebyshev threshold
                if np.any(z_score > theta_cheb):
                    end_time = time.time() - start_time
                    print(f"\n Stage 2: Anomaly Detection |  Max Z-score: {np.max(z_score):.4f}  | Time: {end_time:.4f} seconds \n")
                else:
                    end_time = time.time() - start_time
                    print(f"\n Stage 2:  Everything is normal | Time: {end_time:.3f} seconds \n")

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
    except Exception as e:
        print(e)
        return 
    
    # compute Chebyshev threshold
    theta_cheb = nl.theta_cheb(0.01)
    
    # training window size
    w_train = 20 

    # Thread 1: Sniffer 
    t_sniff = threading.Thread(target=sniffer_worker, args=(5, 'en0'), daemon=True)

    # Thread 2: Analyzer 
    t_analyze = threading.Thread(
        target=analyzer_worker, 
        args=(w_train, sigma_train, mean_train, theta_cheb, model, scaler), 
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