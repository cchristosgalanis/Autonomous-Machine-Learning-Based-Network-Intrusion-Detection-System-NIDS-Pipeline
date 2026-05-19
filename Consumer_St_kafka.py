from confluent_kafka import Consumer, KafkaError
import json
import numpy as np
import time
from collections import deque
import stateful
import os
import csv # Προσθήκη για καταγραφή στο CSV

def Consumer_func():

#-------------------------------------------------------------------------------------------------

    # parameters and buffers initialization
    w_train = 20 

    # sliding window buffer 
    buffer = deque(maxlen=150) 
    
#-------------------------------------------------------------------------------------------------
    
    # kafka consumer conf
    conf = {
        'bootstrap.servers': os.getenv('KAFKA_BROKER', 'localhost:9092'),
        'group.id': 'stealth-analyzer-group',
        'auto.offset.reset': 'latest',
        'enable.auto.commit': True
    }

    consumer = Consumer(conf)
    topic_name = 'live-network-flows'
    consumer.subscribe([topic_name])

    # state machine variables
    under_attack = False
    attack_time = 0

    print(f"\n Stealth Analyzer connected to '{topic_name}'. Waiting for data ...\n")
    print("\n Stateful/Stealth Analysis initialized \n")

    # --- ΠΡΟΣΘΗΚΗ: Αρχικοποίηση αρχείου CSV για το γράφημα των Stealth Attacks ---
    csv_file_path = "logs/stealth_metrics.csv"
    os.makedirs("logs", exist_ok=True)
    
    # Αν το αρχείο δεν υπάρχει, το φτιάχνουμε και βάζουμε την πρώτη γραμμή (headers)
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "IAT_mean", "IAT_std", "SA_ratio", "Is_Attack"])
    # -----------------------------------------------------------------------------

#-------------------------------------------------------------------------------------------------

    try:
        # main consumer loop
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
            
            # convert dict to numpy array row -> [U_ports, SA_ratio, IAT_mean, IAT_std]
            current_row = np.array([flow_data['U_ports'], flow_data['SA_ratio'], flow_data['IAT_mean'], flow_data['IAT_std']])
            
            # update buffer 
            buffer.append(current_row)

            # ML and stateful analysis
            if len(buffer) >= w_train:
                current_data = np.array(list(buffer))
                
                # select current analysis window
                analysis_batch = current_data[-w_train:]
                actual_w = len(analysis_batch)
                
                U_ports = analysis_batch[:, 0]
                SA_ratio = analysis_batch[:, 1]
                IAT_mean = analysis_batch[:, 2]
                IAT_std = analysis_batch[:, 3]

                start_time = time.time()
                # compute stateful check for stealth attacks (Slowloris)
                anomaly_detected = stateful.slowloris_detection(IAT_mean, IAT_std, SA_ratio)
                
                if anomaly_detected:
                    if not under_attack:
                        end_time = time.time() - start_time
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # update state
                        under_attack = True
                        attack_time = time.time()
                        anomaly_type = "Slowloris (Stealth Attack)"
                        
                        # log alert
                        log_message = f"[{timestamp}] Anomaly: {anomaly_type} started! | IAT Mean: {IAT_mean[-1]:.4f}, IAT Std: {IAT_std[-1]:.4f}\n"
                        os.makedirs("logs", exist_ok=True)
                        with open("logs/ids_alerts.log", "a", encoding="utf-8") as log_file:
                            log_file.write(log_message)
                        
                        print(f"\n [!!!] Attack Started | Type: {anomaly_type} | Time: {end_time:.8f} seconds")
                    else:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"[{timestamp}] ... stealth attack is still ongoing ...")
                else:
                    if under_attack:
                        under_attack = False
                        attack_duration = time.time() - attack_time
                        print(f"\n Attack Ended | Duration: {attack_duration:.2f} seconds \n")
                    else:
                        end_time = time.time() - start_time
                        print(f" Traffic is normal | Time: {end_time:.8f} s")

                # --- ΠΡΟΣΘΗΚΗ: ΣΥΝΕΧΗΣ ΚΑΤΑΓΡΑΦΗ ΔΕΔΟΜΕΝΩΝ ΣΤΟ CSV ΓΙΑ ΤΟ ΓΡΑΦΗΜΑ ---
                # Καταγράφεται στο τέλος κάθε batch τις πιο πρόσφατες μετρικές [-1]
                current_plot_time = time.time()
                current_alert_status = 1 if under_attack else 0
                
                with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([current_plot_time, IAT_mean[-1], IAT_std[-1], SA_ratio[-1], current_alert_status])
                # ----------------------------------------------------------------------

#----------------------------------------------------------------------------------------------------------------------------

    except KeyboardInterrupt:
        print("\n Stop Analyzer ... \n")
    finally:
        consumer.close()

if __name__ == "__main__":
    Consumer_func()