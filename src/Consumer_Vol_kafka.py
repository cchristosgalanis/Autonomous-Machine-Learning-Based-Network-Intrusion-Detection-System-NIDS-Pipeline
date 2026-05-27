from confluent_kafka import Consumer, Producer
import json
import numpy as np
import time
from collections import deque
import non_linear as nl
import os  
import csv 

def delivery_report(err, msg):
    if err is not None:
        print(f" [Kafka Error] Message delivery failed: {err}")

def Consumer_func():
    print("\n Loading model, scaler, and training metrics...")
    model, scaler = nl.load_model_and_scaler()
    
    if model is None or scaler is None:
        print("Error: Model or scaler could not be loaded. Exiting.")
        return
    
    w_train = 20 
    ip_buffers = {}
    ip_residual_memories = {}

    broker = os.getenv('KAFKA_BROKER', 'localhost:9092')
    conf_consumer = {
        'bootstrap.servers': broker,
        'group.id': 'ml-analyzer-group-1',
        'auto.offset.reset': 'latest',
        'enable.auto.commit': True
    }

    conf_producer = {'bootstrap.servers': broker}
    
    consumer = Consumer(conf_consumer)
    producer = Producer(conf_producer)
    
    input_topic = 'live-network-flows'
    output_topic = 'unified-features-topic'
    
    consumer.subscribe([input_topic])

    print(f"\n Volumetric Feature Extractor connected.")
    print(f" Reading from '{input_topic}' -> Writing to '{output_topic}'\n")

    csv_file_path = "logs/residual_metrics.csv"
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Source_IP", "Residual", "Velocity", "Acceleration"])

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None or msg.error(): continue

            raw_bytes = msg.value()
            flow_data = json.loads(raw_bytes.decode('utf-8'))
            
            src_ip = flow_data.get('Source_IP', 'Unknown')
            current_row = np.array([flow_data['T_vol'], flow_data['N_req'], flow_data['S_len']])
            
            if src_ip not in ip_buffers:
                ip_buffers[src_ip] = deque(maxlen=150)
                ip_residual_memories[src_ip] = deque(maxlen=3)

            ip_buffers[src_ip].append(current_row)

            log_velocity, log_acceleration, current_residual = 0.0, 0.0, 0.0

            if len(ip_buffers[src_ip]) >= w_train:
                analysis_batch = np.array(list(ip_buffers[src_ip]))[-w_train:]
                actual_w = len(analysis_batch)
                
                T_vol, N_req, S_len = analysis_batch[:, 0], analysis_batch[:, 1], analysis_batch[:, 2]
                entropy_res_array = nl.entropy_based_stab_check(T_vol, N_req, S_len, actual_w, model, scaler)
                current_residual = float(np.mean(entropy_res_array))
                ip_residual_memories[src_ip].append(current_residual)
                
                if len(ip_residual_memories[src_ip]) >= 2:
                    vel, acc = nl.compute_kinematic(list(ip_residual_memories[src_ip]))
                    # Παίρνουμε τις απόλυτες τιμές όπως στο training (γραμμή 46 στο feature_extraction.py)
                    log_velocity = abs(vel)
                    log_acceleration = abs(acc)

            current_plot_time = time.time()
            
            # csv logging
            with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([current_plot_time, src_ip, current_residual, log_velocity, log_acceleration])

            # kafka payload για το unified-features topic
            feature_payload = {
                "source_ip": src_ip,
                "analyzer": "volumetric",
                "metrics": {
                    "velocity": log_velocity,
                    "acceleration": log_acceleration
                }
            }
            
            producer.produce(
                topic=output_topic, 
                key=src_ip.encode('utf-8'), 
                value=json.dumps(feature_payload).encode('utf-8'),
                callback=delivery_report
            )
            producer.poll(0)

    except KeyboardInterrupt:
        print("\n Stop Extractor ... \n")
    finally:
        consumer.close()
        producer.flush()

if __name__ == "__main__":
    Consumer_func()