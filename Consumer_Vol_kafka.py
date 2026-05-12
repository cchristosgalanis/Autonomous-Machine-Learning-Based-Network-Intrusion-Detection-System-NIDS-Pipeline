from confluent_kafka import Consumer, KafkaError
import json
import numpy as np
import time
from collections import deque
import non_linear as nl

def Consumer_func():

#-------------------------------------------------------------------------------------------------

    # --- model & metrics initialization ---
    print("\n Loading model, scaler, and training metrics...")
    model, scaler = nl.load_model_and_scaler()
    
    if model is None or scaler is None:
        print("Error: Model or scaler could not be loaded. Exiting.")
        return
    
#-------------------------------------------------------------------------------------------------

    # loading parameters & raw metrics for signature analysis
    try:
        params = np.load('train_metrics_feature_expansion.npz')
        raw_mean = params['raw_mean']
        raw_std = params['raw_std']
    except Exception as e:
        print(f"Error loading training metrics: {e}")
        return 

    # --- parameters and buffers initialization ---
    w_train = 20 
    kinematic_threshold = 0.8 # or 1.0
    velocity_threshold = 0.65  # threshold for 1st derivative 
    
    # minimum packets threshold to consider an anomaly valid (to avoid false positives from kinematic spikes during very low traffic periods)
    min_packets_threshold = 50 #can be tuned based on training data or set to 0 to disable this check
    
    # sliding window buffer 
    buffer = deque(maxlen=150) 
    # kinematic memory buffer 
    residual_memory = deque(maxlen=3) 

#-------------------------------------------------------------------------------------------------

    # --- kafka consumer conf ---
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'ml-analyzer-group-1',
        'auto.offset.reset': 'latest',
        'enable.auto.commit': True
    }

    consumer = Consumer(conf)
    topic_name = 'live-network-flows'
    consumer.subscribe([topic_name])

    # state machine variables
    under_attack = False
    attack_time = 0

    print(f"\n Analyzer connected to '{topic_name}'. Waiting for data ...\n")
    print("\n Linear Regression Model & Kinematic Analyzer initialized \n")

#-------------------------------------------------------------------------------------------------

    try:
        # --- main consumer loop ---
        while True:
            # poll Kafka for new messages
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                print(f"\n Consumer Error: {msg.error()}")
                continue

            # decode json data
            raw_bytes = msg.value()
            json_string = raw_bytes.decode('utf-8')
            flow_data = json.loads(json_string)
            
            # convert dict to numpy array row -> [T_vol, N_req, S_len]
            current_row = np.array([flow_data['T_vol'], flow_data['N_req'], flow_data['S_len']])
            
            # update buffer 
            buffer.append(current_row)

            # --- ML and derivatives ---
            if len(buffer) >= w_train:
                current_data = np.array(list(buffer))
                
                # select current analysis window
                analysis_batch = current_data[-w_train:]
                actual_w = len(analysis_batch)
                
                T_vol = analysis_batch[:, 0]
                N_req = analysis_batch[:, 1]
                S_len = analysis_batch[:, 2]

                # compute residuals using Renyi entropy and Linear Regression
                entropy_res_array = nl.entropy_based_stab_check(T_vol, N_req, S_len, actual_w, model, scaler)
                
                # extract current residual error
                current_residual = float(np.mean(entropy_res_array))
                residual_memory.append(current_residual)
                
                if not under_attack:
                    # benign (checking for velocity or acceleration spike)
                    
                    if len(residual_memory) >= 2:
                        start_time = time.time()
                    
                        # calculate 1st (velocity) and 2nd (acceleration) derivatives
                        velocity, acceleration = nl.compute_kinematic(list(residual_memory))

                        #get current packet count for the last second (assuming T_vol is packets per second)
                        current_packets = current_data[-1][1]

                        kinematic_spike = abs(velocity) > velocity_threshold or (len(residual_memory) == 3 and abs(acceleration) > kinematic_threshold)

                        if kinematic_spike and (current_packets > min_packets_threshold):
                            end_time = time.time() - start_time
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            
                            # update state
                            under_attack = True
                            attack_time = time.time()

                            # check signature to classify the anomaly
                            anomaly_type = nl.signature_analysis(current_data[-1:], raw_mean, raw_std)
                        
                            log_message = f"[{timestamp}] Anomaly: {anomaly_type} started! | Vel: {velocity:.4f}, Acc: {acceleration:.4f}\n"
                        
                            # log alert to file
                            with open("ids_alerts.log", "a", encoding="utf-8") as log_file:
                                log_file.write(log_message)
                            
                            print(f"\n [!!!] Attack Started | Type: {anomaly_type} | Time: {end_time:.8f} seconds")
                        
                            # clear memory to prevent alert flooding
                            residual_memory.clear()
                        
                        else:
                            end_time = time.time() - start_time
                            print(f" Kinematics stable (Vel: {velocity:.4f}, Acc: {acceleration:.4f}) | Time: {end_time:.8f} s")
                
                    else:
                        # wait for memory to fill 
                        print(f" Warming up kinematic memory... ({len(residual_memory)}/2 steps for Vel, 3 for Acc)")

                else:
                    # under_attack -> True
                    # evaluate current traffic using signature analysis
                    current_status = nl.signature_analysis(current_data[-1:], raw_mean, raw_std)
                    
                    if current_status == "Malicious Attack (Potential Flood)":
                        # attack is sustaining
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        log_msg_ongoing = f"[{timestamp}]  Attack Ongoing: {current_status} is still active...\n"

                        with open("ids_alerts.log", "a", encoding="utf-8") as log_file:
                            log_file.write(log_msg_ongoing)
                            
                        print(f"[{timestamp}] ... attack is still ongoing ...")

                    else:
                        # traffic normalized
                        under_attack = False
                        attack_duration = time.time() - attack_time
                        
                        print(f"\n Attack Ended | Duration: {attack_duration:.2f} seconds \n")
                        
                        # clear memory to avoid false kinematics from the sudden drop
                        residual_memory.clear()

#----------------------------------------------------------------------------------------------------------------------------

    except KeyboardInterrupt:
        print("\n Stop Analyzer ... \n")
    finally:
        consumer.close()

if __name__ == "__main__":
    Consumer_func()