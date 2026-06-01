from confluent_kafka import Consumer
import json
import numpy as np
import time
import os
import non_linear as nl

def Consumer_NN_func():
    HOST_IP = os.getenv('HOST_IP', '127.0.0.1')
    
    # Define paths for hot-reloading
    model_path = os.path.join("models", "mlp_model.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")

    print("\n [AI Engine] Loading Initial Neural Network and Scaler...")
    model, scaler = nl.load_nn_model_and_scaler()
    
    if model is None or scaler is None:
        print("Error: Model or scaler could not be loaded. Exiting.")
        return

    # Track the last time the model file was modified on disk
    last_model_timestamp = os.path.getmtime(model_path) if os.path.exists(model_path) else 0

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

    WINDOW_INTERVAL = 10.0
    last_evaluation_time = time.time()
    aggregate_buffer = {}

    os.makedirs("logs", exist_ok=True)
    master_log_path = "logs/master_traffic_records.jsonl"

    try:
        while True:
            msg = consumer.poll(timeout=0.5)
            current_time = time.time()

            if msg is not None and not msg.error():
                try:
                    payload = json.loads(msg.value().decode('utf-8'))
                    #change on this line, in order to avoid get port number instead of IP address, since some analyzers (like spatial) don't have IP as key
                    entity_id = payload.get('source_ip','Unknown')
                    if entity_id == 'Unknown':
                        continue

                    analyzer = payload.get('analyzer')
                    metrics = payload.get('metrics', {})

                    if entity_id not in aggregate_buffer:
                        aggregate_buffer[entity_id] = {
                            'volumetric': [], 'stealth': [], 'spatial': [], 'dns': []
                        }
                    
                    if analyzer in aggregate_buffer[entity_id]:
                        aggregate_buffer[entity_id][analyzer].append(metrics)

                except Exception as e:
                    print(f" [!] JSON parsing error: {e}")

            if current_time - last_evaluation_time >= WINDOW_INTERVAL:
                
                # --- HOT RELOAD LOGIC ---
                # Check if retraining service dropped a new model before evaluating
                if os.path.exists(model_path):
                    current_model_timestamp = os.path.getmtime(model_path)
                    
                    if current_model_timestamp > last_model_timestamp:
                        print("\n [!] Model update detected on disk! Hot-reloading weights...")
                        try:
                            # Load the new weights and scaler
                            new_model, new_scaler = nl.load_nn_model_and_scaler()
                            if new_model is not None and new_scaler is not None:
                                model = new_model
                                scaler = new_scaler
                                last_model_timestamp = current_model_timestamp
                                print(" [SUCCESS] New Neural Network weights loaded seamlessly.")
                            else:
                                print(" [!] Failed to load new weights. Keeping old model in memory.")
                        except Exception as e:
                            print(f" [!] Error during hot-reload: {e}")
                # ------------------------

                if len(aggregate_buffer) > 0:
                    X_batch = []
                    entities_batch = []
                    
                    for entity, data in aggregate_buffer.items():
                        vel = max([v.get('velocity', 0.0) for v in data['volumetric']]) if data['volumetric'] else 0.0
                        acc = max([v.get('acceleration', 0.0) for v in data['volumetric']]) if data['volumetric'] else 0.0
                        iat = float(np.mean([s.get('iat_mean', 0.0) for s in data['stealth']])) if data['stealth'] else 0.0
                        sa = max([s.get('sa_ratio', 0.0) for s in data['stealth']]) if data['stealth'] else 0.0
                        jac = max([s.get('jaccard_score', 0.0) for s in data['spatial']]) if data['spatial'] else 0.0
                        ent = max([d.get('entropy_shannon', 0.0) for d in data['dns']]) if data['dns'] else 0.0
                        
                        X_batch.append([vel, acc, iat, sa, jac, ent])
                        entities_batch.append(entity)

                    X_np = np.array(X_batch)
                    X_scaled = scaler.transform(X_np)
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)[:, 1] 

                    print(f"\n --- [AI Engine] Window Evaluated: {len(entities_batch)} unique IPs ---")

                    with open(master_log_path, "a", encoding="utf-8") as master_log, \
                         open("logs/nids_final_alerts.log", "a", encoding="utf-8") as alert_log:
                        
                        for idx, prob in enumerate(probabilities):
                            ip = entities_batch[idx]
                            pred = int(predictions[idx])
                            
                            record = {
                                "timestamp": current_time,
                                "entity_id": ip,
                                "features": X_batch[idx],
                                "prediction": pred,
                                "probability": float(prob),
                                "is_host_ip": (ip == HOST_IP)
                            }
                            master_log.write(json.dumps(record) + "\n")
                            
                            if pred == 1:
                                timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
                                alert_msg = f"[{timestamp_str}] ALERT: Malicious Traffic from {ip} (Prob: {prob:.2%})\n"
                                print(alert_msg.strip())
                                alert_log.write(alert_msg)
                
                aggregate_buffer.clear()
                last_evaluation_time = current_time

    except KeyboardInterrupt:
        print("\n [AI Engine] Shutting down... \n")
    finally:
        consumer.close()

if __name__ == "__main__":
    Consumer_NN_func()