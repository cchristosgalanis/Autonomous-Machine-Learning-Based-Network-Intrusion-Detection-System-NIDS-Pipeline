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


# Consumer Thread: Kinematic Analysis & Detection
# Optimized to receive pre-loaded model, scaler, and kinematic threshold
def analyzer_worker(w_train, kinematic_threshold, model, scaler):
    # Sliding window buffer (traffic history)
    buffer = deque(maxlen=150) 
    
    # Kinematic memory buffer (keeps exactly the last 3 residual steps: t-2, t-1, t)
    residual_memory = deque(maxlen=3) 
    
    print("\n Linear Regression Model & Kinematic Analyzer initialized \n")
    
    while True:
        # Get new batch of data from the sniffer
        new_df = data_queue.get()
        
        # Update the buffer (past + present)
        for _, row in new_df.iterrows():
            buffer.append(row.values)
        
        # Start analysis if buffer has the minimum window size
        if len(buffer) >= w_train:
            current_data = np.array(list(buffer))
            
            # Select the current analysis window
            analysis_batch = current_data[-w_train:]
            actual_w = len(analysis_batch) # Get size of the slice
            
            T_vol = analysis_batch[:, 0]
            N_req = analysis_batch[:, 1]
            S_len = analysis_batch[:, 2]

            # --- Entropy & ML Prediction ---
            # Compute residuals using Renyi entropy and Linear Regression
            entropy_res_array = nl.entropy_based_stab_check(T_vol, N_req, S_len, actual_w, model, scaler)
            
            # Extract the current residual value (scalar number)
            # Taking the mean of the array to get a single representation of the current window's error
            current_residual = float(np.mean(entropy_res_array))
            
            # Update the kinematic memory with the latest residual
            residual_memory.append(current_residual)
            
            # --- Kinematic Stability Check (1st and 2nd Derivative) ---
            
            # Check if we have gathered enough steps (t-2, t-1, t)
            if len(residual_memory) == 3:
                start_time = time.time()
                
                # Calculate the 2nd derivative (acceleration) of the error
                acceleration = nl.compute_kinematic(list(residual_memory))

                print(f"\n[DEBUG] Current Residual: {current_residual:.4f}")
                print(f"[DEBUG] Acceleration:     {acceleration:.4f}")
                print(f"[DEBUG] Threshold:        {kinematic_threshold}")
                print("-" * 40)
                
                # Dynamic Decision based on the acceleration spike
                if abs(acceleration) > kinematic_threshold:
                    end_time = time.time() - start_time
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    log_message = f"[{timestamp}] ALERT: Abrupt Kinematic Anomaly (DDoS/Scan) | Acceleration Spike: {acceleration:.4f}\n"
                    
                    with open("ids_alerts.log", "a", encoding="utf-8") as log_file:
                        log_file.write(log_message)
                        
                    print(f"!!! Anomaly Detected | Type: Kinematic Shift | Time: {end_time:.8f} seconds")
                    
                    # Clear kinematic memory to prevent alert flooding from the same spike
                    residual_memory.clear()
                    
                else:
                    end_time = time.time() - start_time
                    print(f" Kinematics are stable | Everything is normal | Time: {end_time:.8f} seconds \n")
            
            else:
                # Waiting for memory to fill up
                print(f" Warming up kinematic memory... ({len(residual_memory)}/3 steps) \n")
        
        # Signal that the batch processing is complete
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
    kinematic_threshold = 1.0

    # Thread 1: Sniffer 
    t_sniff = threading.Thread(target=sniffer_worker, args=(1, 'en0'), daemon=True)

    # Thread 2: Analyzer 
    t_analyze = threading.Thread(
        target=analyzer_worker, 
        args=(w_train, kinematic_threshold,model,scaler), 
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