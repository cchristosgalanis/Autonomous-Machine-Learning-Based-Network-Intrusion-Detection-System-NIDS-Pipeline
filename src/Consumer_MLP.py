from confluent_kafka import Consumer
import json
import numpy as np
import time
import os
import non_linear as nl
import psycopg2
from psycopg2 import extras
import datetime

def rotate_log_file(filepath, max_lines=1000):
    if not os.path.exists(filepath):
        return
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if len(lines) > max_lines:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(lines[-max_lines:])
    except Exception as e:
        print(f"\n Error rotating log {filepath}: {e}")

# main function for the consumer that runs the neural network inference and handles hot-reloading of the model weights
def Consumer_NN_func():
    HOST_IP = os.getenv('HOST_IP', '127.0.0.1')
    
    # database credentials
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASS = os.getenv('DB_PASS', 'yourpassword')
    DB_NAME = os.getenv('DB_NAME', 'postgres')
    
    model_path = os.path.join("models", "mlp_model1.pkl")
    scaler_path = os.path.join("models", "scaler1.pkl")

    print("\n --AI Engine-- Loading Initial Neural Network and Scaler... \n")
    model, scaler = nl.load_nn_model_and_scaler()
    
    if model is None or scaler is None:
        print("Error: Model or scaler could not be loaded. Exiting.")
        return

    # kafka consumer configuration
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

    # initial log message
    print(f"\n --AI Engine-- Listening to '{input_topic}' for 10-second windows...\n")

    WINDOW_INTERVAL = 10.0
    last_evaluation_time = time.time()
    aggregate_buffer = {}

    os.makedirs("logs", exist_ok=True)
    alerts_log_path = "logs/nids_final_alerts.log"

    try:
        while True:
            msg = consumer.poll(timeout=0.5)
            current_time = time.time()

            if msg is not None and not msg.error():
                try:
                    payload = json.loads(msg.value().decode('utf-8'))
                    entity_id = payload.get('source_ip','Unknown')
                    if entity_id == 'Unknown':
                        continue

                    analyzer = payload.get('analyzer', 'unknown')
                    metrics = payload.get('metrics', {})

                    if entity_id not in aggregate_buffer:
                        aggregate_buffer[entity_id] = {
                            'volumetric': [], 'stealth': [], 'spatial': [], 'dns': []
                        }
                    
                    if analyzer in aggregate_buffer[entity_id]:
                        aggregate_buffer[entity_id][analyzer].append(metrics)

                except Exception as e:
                    print(f"\n JSON parsing error: {e}")

            if current_time - last_evaluation_time >= WINDOW_INTERVAL:
                
                # --- hot reload logic ---
                flag_path = "models/reload_flag.txt"
                if os.path.exists(flag_path):
                    print("\n Model update flag detected! Hot-reloading weights...")
                    try:
                        new_model, new_scaler = nl.load_nn_model_and_scaler()
                        if new_model is not None and new_scaler is not None:
                            model = new_model
                            scaler = new_scaler
                            print("\n New Neural Network weights loaded seamlessly.")
                            os.remove(flag_path)
                        else:
                            print("\n Failed to load new weights. Keeping old model in memory.")
                    except Exception as e:
                        print(f"\n Error during hot-reload: {e}")

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

                    # write results to database and log alerts for malicious predictions
                    db_records = []
                    
                    with open(alerts_log_path, "a", encoding="utf-8") as alert_log:
                        for idx, prob in enumerate(probabilities):
                            ip = entities_batch[idx]
                            pred = int(predictions[idx])
                            features = X_batch[idx]
                            is_host = (ip == HOST_IP)
                            dt_time = datetime.datetime.fromtimestamp(current_time)
                            
                            # Heuristic override for completely benign/idle/quiet traffic to bypass model's zero-traffic bias
                            vel, acc, iat, sa, jac, ent = features
                            if vel == 0.0 and acc == 0.0 and iat == 0.0 and sa <= 1.2 and jac == 0.0 and ent < 2.5:
                                pred = 0
                                prob = 0.0
                            else:
                                # Only classify as attack (pred = 1) if the anomaly probability is high (>= 0.85)
                                pred = 1 if prob >= 0.85 else 0
                            
                            # create tuple for sql database insertion
                            db_records.append((
                                dt_time, ip, 'aggregated', is_host,
                                float(features[0]), float(features[1]), float(features[2]),
                                float(features[3]), float(features[4]), float(features[5]),
                                float(prob), pred
                            ))
                            
                            if pred == 1 and prob >= 0.90:
                                timestamp_str = dt_time.strftime("%Y-%m-%d %H:%M:%S")
                                alert_msg = f"[{timestamp_str}] ALERT: Malicious Traffic from {ip} (Prob: {prob:.2%})\n"
                                print(alert_msg.strip())
                                alert_log.write(alert_msg)
                    
                    # write to database in batches for efficiency
                    if db_records:
                        try:
                            conn = psycopg2.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, dbname=DB_NAME)
                            cursor = conn.cursor()
                            
                            insert_query = """
                                INSERT INTO network_traffic_events (
                                    time, source_ip, analyzer, is_host_ip, 
                                    velocity, acceleration, iat_mean, sa_ratio, jaccard_score, entropy_shannon, 
                                    probability, prediction
                                ) VALUES %s
                            """
                            extras.execute_values(cursor, insert_query, db_records)
                            conn.commit()
                            
                            cursor.close()
                            conn.close()
                        except Exception as db_err:
                            print(f"\n Database Insert Error: {db_err}")
                    
                    rotate_log_file(alerts_log_path, max_lines=1000)
                
                aggregate_buffer.clear()
                last_evaluation_time = current_time

    except KeyboardInterrupt:
        print("\n --AI Engine-- Shutting down... \n")
    finally:
        consumer.close()

if __name__ == "__main__":
    Consumer_NN_func()