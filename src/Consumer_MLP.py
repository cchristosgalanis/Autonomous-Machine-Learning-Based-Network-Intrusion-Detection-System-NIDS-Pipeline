from confluent_kafka import Consumer
import json
import numpy as np
import time
import joblib
import os
import non_linear as nl

def Consumer_NN_func():
    HOST_IP = os.getenv('HOST_IP', '127.0.0.1')

    print("\n [AI Engine] Loading Neural Network and Scaler...")
    
    model, scaler = nl.load_nn_model_and_scaler()

    if model is None or scaler is None:
        print("Error: Model or scaler could not be loaded. Exiting.")
        return

    # kafka consumer configuration
    broker = os.getenv('KAFKA_BROKER', 'localhost:9092')
    conf_consumer = {
        'bootstrap.servers': broker,
        'group.id': 'neural-network-group',
        'auto.offset.reset': 'latest',
        'enable.auto.commit': True
    }
    
    consumer = Consumer(conf_consumer)
    input_topic = 'unified-features-topic'
    consumer.subscribe([input_topic])

    print(f"\n [AI Engine] Listening to '{input_topic}' for 10-second windows...\n")

    # variables
    WINDOW_INTERVAL = 10.0
    last_evaluation_time = time.time()
    
    # dictionary
    # { '192.168.1.5': { 'volumetric': [], 'stealth': [], 'spatial': [], 'dns': [] } }
    aggregate_buffer = {}

    try:
        while True:
            # poll
            msg = consumer.poll(timeout=0.5)
            current_time = time.time()

            if msg is not None and not msg.error():
                try:
                    payload = json.loads(msg.value().decode('utf-8'))
                    
                    entity_id = payload.get('source_ip', payload.get('target_port', 'Unknown'))
                    analyzer = payload.get('analyzer')
                    metrics = payload.get('metrics', {})

                    # initialize
                    if entity_id not in aggregate_buffer:
                        aggregate_buffer[entity_id] = {
                            'volumetric': [], 'stealth': [], 'spatial': [], 'dns': []
                        }
                    
                    # adding new metrics
                    if analyzer in aggregate_buffer[entity_id]:
                        aggregate_buffer[entity_id][analyzer].append(metrics)

                except Exception as e:
                    print(f" [!] JSON parsing error: {e}")

            # time > window interval -> evaluate
            if current_time - last_evaluation_time >= WINDOW_INTERVAL:
                
                if len(aggregate_buffer) > 0:
                    X_batch = []
                    entities_batch = []
                    
                    # --- feature Alignment & Imputation (Συμπλήρωση Κενών) ---
                    for entity, data in aggregate_buffer.items():
                        
                        # volumetric
                        vol = data['volumetric']
                        vel = max([v['velocity'] for v in vol]) if vol else 0.0
                        acc = max([v['acceleration'] for v in vol]) if vol else 0.0
                        
                        # stealth
                        stl = data['stealth']
                        iat = float(np.mean([s['iat_mean'] for s in stl])) if stl else 0.0
                        sa = max([s['sa_ratio'] for s in stl]) if stl else 0.0
                        
                        # spatial (Botnet)
                        spa = data['spatial']
                        jac = max([s['jaccard_score'] for s in spa]) if spa else 0.0
                        
                        # DNS (DGA)
                        dns_data = data['dns']
                        ent = max([d['entropy_shannon'] for d in dns_data]) if dns_data else 0.0
                        
                        # 6D vector
                        vector = [vel, acc, iat, sa, jac, ent] # velocity, acceleration, iat_mean, sa_ratio, jaccard_score, dns_entropy
                        
                        X_batch.append(vector)
                        entities_batch.append(entity)

                    # --- Batch Prediction 
                    X_np = np.array(X_batch)
                    X_scaled = scaler.transform(X_np)
                    
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)[:, 1] # Πιθανότητα για κλάση 1 (Attack)

                    print(f"\n --- [AI Engine] Window Evaluated: {len(entities_batch)} unique IPs ---")

                    # results
                    for idx, pred in enumerate(predictions):
                        ip = entities_batch[idx]
                        prob = probabilities[idx]
                        
                        if pred == 1:
                            # attack
                            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
                            alert_msg = f"[{timestamp_str}] CRITICAL ALERT: Neural Network detected Malicious Traffic from {ip} (Confidence: {prob:.2%})\n"
                            print(alert_msg.strip())
                            
                            # final NDS alert logging
                            with open("logs/nids_final_alerts.log", "a", encoding="utf-8") as f:
                                f.write(alert_msg)
                        else:
                            # benign
                            pass 
                
                # clear the buffer for the next 10-minute window and reset the timer
                aggregate_buffer.clear()
                last_evaluation_time = current_time

    except KeyboardInterrupt:
        print("\n [AI Engine] Shutting down... \n")
    finally:
        consumer.close()

if __name__ == "__main__":
    Consumer_NN_func()