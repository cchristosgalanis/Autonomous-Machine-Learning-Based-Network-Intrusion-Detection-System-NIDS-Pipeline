from confluent_kafka import Consumer, Producer
import json
import time
import os 
import csv 

def delivery_report(err, msg):
    if err is not None:
        print(f" [Kafka Error] Message delivery failed: {err}")

def Consumer_Stealth_func():
    broker = os.getenv('KAFKA_BROKER', 'localhost:9092')
    
    conf_consumer = {
        'bootstrap.servers': broker,
        'group.id': 'stealth-analyzer-group',
        'auto.offset.reset': 'latest',
        'enable.auto.commit': True
    }
    conf_producer = {'bootstrap.servers': broker}

    consumer = Consumer(conf_consumer)
    producer = Producer(conf_producer)
    
    input_topic = 'live-network-flows'
    output_topic = 'unified-features-topic'
    
    consumer.subscribe([input_topic])

    print(f"\n Stealth Feature Extractor connected.")
    print(f" Reading from '{input_topic}' -> Writing to '{output_topic}'\n")

    csv_file_path = "logs/stealth_metrics.csv"
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Source_IP", "IAT_mean", "SA_ratio"])

    try:
        while True:
            msg = consumer.poll(timeout=1.0)
            if msg is None or msg.error(): continue

            raw_bytes = msg.value()
            flow_data = json.loads(raw_bytes.decode('utf-8'))

            src_ip = flow_data.get('Source_IP', 'Unknown')
            try:
                sa_ratio = float(flow_data.get('SA_ratio', 0.0))
                iat_mean = float(flow_data.get('IAT_mean', 0.0))
            except KeyError:
                continue

            current_plot_time = time.time() 
            
            # csv logging
            with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([current_plot_time, src_ip, iat_mean, sa_ratio])

            # kafka payload για το unified-features topic
            feature_payload = {
                "source_ip": src_ip,
                "analyzer": "stealth",
                "metrics": {
                    "iat_mean": iat_mean,
                    "sa_ratio": sa_ratio
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
        print("\n Shutting down... \n")
    finally:
        consumer.close()
        producer.flush()

if __name__ == "__main__":
    Consumer_Stealth_func()