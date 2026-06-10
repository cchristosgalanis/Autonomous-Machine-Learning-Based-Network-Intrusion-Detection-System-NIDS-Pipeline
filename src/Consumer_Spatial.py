from confluent_kafka import Consumer, Producer
import json
import numpy as np
import os
import time  
import csv  
import non_linear as nl

def delivery_report(err, msg):
    if err is not None: print(f" [Kafka Error] Message delivery failed: {err}")

# function for the consumer that listens to the Kafka topic, performs spatial and DNS analysis, and produces unified features for the MLP
def Consumer_Spatial_func():
    broker = os.getenv('KAFKA_BROKER', 'localhost:9092')

    #kafka consumer and producer configuration
    conf_consumer = {
        'bootstrap.servers': broker,
        'group.id': 'spatial-analyzer-group',
        'auto.offset.reset': 'latest',
        'enable.auto.commit': True
    }
    conf_producer = {'bootstrap.servers': broker}

    consumer = Consumer(conf_consumer)
    producer = Producer(conf_producer)

    spatial_topic = 'spatial-minhash-signatures'
    dns_topic = 'dns-queries'
    output_topic = 'unified-features-topic'

    consumer.subscribe([spatial_topic, dns_topic])

    print(f"\n Spatial & DNS Feature Extractor connected.")
    print(f" Writing to '{output_topic}'...\n")

    # variables for spatial analysis
    prev_signatures = {}
    JACCARD_THRESHOLD = 0.70

    os.makedirs("logs", exist_ok=True)
    spatial_csv_path = "logs/spatial_metrics.csv"
    dns_csv_path = "logs/dns_metrics.csv"
    
    if not os.path.exists(spatial_csv_path):
        with open(spatial_csv_path, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["Timestamp", "Target_Port", "Jaccard_Score"])
            
    if not os.path.exists(dns_csv_path):
        with open(dns_csv_path, mode='w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(["Timestamp", "Source_IP", "Domain", "Entropy"])

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None or msg.error(): continue

            curr_topic = msg.topic()
            try:
                payload = json.loads(msg.value().decode('utf-8'))
            except Exception as e: continue
            
            current_plot_time = time.time()

            # --- spatial / botnet ---
            if curr_topic == spatial_topic:
                target_port = payload.get('target_port')
                k = payload.get('k_size', 256)
                sig_list = payload.get('signature', [])
                source_ips = payload.get('source_ips', [])
                if not target_port or not sig_list: continue
                    
                signature = np.array(sig_list)
                jaccard_score = 0.0

                if target_port in prev_signatures:
                    prev_sig = prev_signatures[target_port]
                    # Only calculate meaningful similarity if there is a minimum set of active IPs (avoid single IP FPs)
                    if len(source_ips) >= 3:
                        spatial_analysis = nl.spatial_jaccard(signature, prev_sig, k, JACCARD_THRESHOLD)
                        jaccard_score = spatial_analysis['jaccard_score']
                    else:
                        jaccard_score = 0.0

                    with open(spatial_csv_path, mode='a', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerow([current_plot_time, target_port, jaccard_score])

                    for src_ip in source_ips:
                        feature_payload = {
                            "source_ip": src_ip,
                            "analyzer": "spatial",
                            "metrics": {
                                "jaccard_score": float(jaccard_score)
                            }
                        }
                        producer.produce(
                            topic=output_topic, 
                            key=src_ip.encode('utf-8'),
                            value=json.dumps(feature_payload).encode('utf-8'),
                            callback=delivery_report
                        )
                prev_signatures[target_port] = signature

            # --- DNS / DGA ---
            elif curr_topic == dns_topic:
                for query in payload:
                    domain = query.get('domain', '')
                    src_ip = query.get('source_ip', 'Unknown')
                    if not domain: continue

                    entropy_val = nl.shannon_entropy(domain)
                    
                    with open(dns_csv_path, mode='a', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerow([current_plot_time, src_ip, domain, entropy_val])
                    
                    feature_payload = {
                        "source_ip": src_ip,
                        "analyzer": "dns",
                        "metrics": {
                            "entropy_shannon": float(entropy_val)
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
        print("\n Stopping Spatial Analyzer ... \n")
    finally:
        consumer.close()
        producer.flush()

if __name__ == "__main__":
    Consumer_Spatial_func()