from confluent_kafka import Consumer
import json
import numpy as np
import os
import time  
import csv  
import non_linear as nl

def Consumer_Spatial_func():
    # dynamic broker configuration for docker 
    broker = os.getenv('KAFKA_BROKER', 'localhost:9092')

    conf = {
        'bootstrap.servers': broker,
        'group.id': 'spatial-analyzer-group',
        'auto.offset.reset': 'latest',
        'enable.auto.commit': True
    }

    consumer = Consumer(conf)

    # list with two topics
    spatial_topic = 'spatial-minhash-signatures'
    dns_topic = 'dns-queries'

    consumer.subscribe([spatial_topic, dns_topic])

    print(f"\n Spatial Analyzer connected to '{spatial_topic}' and '{dns_topic}'. Waiting for data ...\n")

    # memory state MinHash
    prev_signatures = {}

    # CONST Variables
    JACCARD_THRESHOLD = 0.70
    ENTROPY_THRESHOLD = 3.5

    # --- initialize logs directory and CSV files ---
    os.makedirs("logs", exist_ok=True)
    
    # CSV for Botnet metrics (Jaccard Score)
    spatial_csv_path = "logs/spatial_metrics.csv"
    if not os.path.exists(spatial_csv_path):
        with open(spatial_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Target_Port", "Jaccard_Score", "Is_Botnet"])

    # CSV for DGA/C2 metrics (Shannon Entropy)
    dns_csv_path = "logs/dns_metrics.csv"
    if not os.path.exists(dns_csv_path):
        with open(dns_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Source_IP", "Domain", "Entropy", "Is_DGA"])

    # --------------------------------------------------------------------------------

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue
            if msg.error():
                print(f"\n Consumer Error: {msg.error()}")
                continue

            curr_topic = msg.topic()
            payload = json.loads(msg.value().decode('utf-8'))

         # --------------------------------------------------------------------------------  
            if curr_topic == spatial_topic:
                target_port = payload['target_port']
                k = payload['k_size']
                signature = np.array(payload['signature'])

                if target_port in prev_signatures:
                    prev_sig = prev_signatures[target_port]

                    spatial_analysis = nl.spatial_jaccard(signature, prev_sig, k, JACCARD_THRESHOLD)

                    # --- log metrics to CSV ---
                    current_plot_time = time.time()
                    is_botnet_num = 1 if spatial_analysis['is_botnet'] else 0
                    with open(spatial_csv_path, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([current_plot_time, target_port, spatial_analysis['jaccard_score'], is_botnet_num])
                    # ----------------------------------------------------------------------------------------------------------

                    if spatial_analysis['status'] != 'idle':
                        print(f"\n [Spatial Alert] Target Port: {target_port} | Jaccard Similarity: {spatial_analysis['jaccard_score']:.2f} | Status: {spatial_analysis['status']} \n")

                        if spatial_analysis['is_botnet']:
                            # --- log alert to centralized ids_alerts.log ---
                            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
                            log_msg = f"[{timestamp_str}] Spatial Anomaly: Coordinated Botnet targeting Port {target_port} (Jaccard: {spatial_analysis['jaccard_score']:.2f})\n"
                            
                            with open("logs/ids_alerts.log", "a", encoding="utf-8") as log_file:
                                log_file.write(log_msg)
                            # -------------------------------------------------------------------------------------------------------------------------------
                            
                            print(f"\n [Botnet Alert] Target Port: {target_port} | Jaccard Similarity: {spatial_analysis['jaccard_score']:.2f} | Status: {spatial_analysis['status']} \n")
                
                prev_signatures[target_port] = signature

        # ----------------------------------------------------------------------------------------------------------------------------------------------------------------

            elif curr_topic == dns_topic:
                for query in payload:
                    domain = query.get('domain', '')
                    src_ip = query.get('source_ip', '') # retrieve source_ip as sent by the Producer

                    dns_analysis = nl.dns_entropy_check(domain, ENTROPY_THRESHOLD)

                    # --- log metrics to CSV ---
                    current_plot_time = time.time()
                    is_dga_num = 1 if dns_analysis['is_DGA'] else 0
                    with open(dns_csv_path, mode='a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([current_plot_time, src_ip, domain, dns_analysis['entropy'], is_dga_num])
                    
                    # ----------------------------------------------------------------------------------------------------------

                    if dns_analysis['is_DGA']:
                        # --- log alert to centralized ids_alerts.log ---
                        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
                        log_msg = f"[{timestamp_str}] DNS Anomaly: DGA/C2 Domain '{domain}' from IP {src_ip} (Entropy: {dns_analysis['entropy']:.2f})\n"
                        
                        with open("logs/ids_alerts.log", "a", encoding="utf-8") as log_file:
                            log_file.write(log_msg)
                        
                        # -------------------------------------------------------------------------------------------------------------------------------
                        
                        print(f"\n [DGA Alert] Source IP: {src_ip} | Domain: {domain} | Entropy: {dns_analysis['entropy']:.2f} | Status: DGA-like \n")
                    else:
                        pass
                
    except KeyboardInterrupt:
        print("\n Stopping Spatial Analyzer ... \n")
    finally:
        consumer.close()

if __name__ == "__main__":
    Consumer_Spatial_func()