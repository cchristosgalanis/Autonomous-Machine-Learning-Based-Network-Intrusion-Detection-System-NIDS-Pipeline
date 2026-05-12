from confluent_kafka import Consumer
import json
import time
import os 
import stateful_functions as sf


def Consumer_Stealth_func():
    # --- Kafka configuration ---
    conf = {
        'bootstrap.servers': os.getenv('KAFKA_BROKER', 'localhost:9092'),
        'group.id': 'stealth-analyzer-group',
        'auto.offset.reset': 'latest',
        'enable.auto.commit': True
    }

    consumer = Consumer(conf)
    topic_name = 'live-network-flows'
    consumer.subscribe([topic_name])

    curr_attack_state = 'Normal'

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue
            if msg.error():
                print(f"\n Consumer Error: {msg.error()}")
                continue

            # decode JSON file
            raw_bytes = msg.value()
            flow_data = json.loads(raw_bytes.decode('utf-8'))

            # extract only stealth features
            try:
                u_ports = int(flow_data.get('U_ports', 0))
                sa_ratio = float(flow_data.get('SA_ratio', 0.0))
                iat_mean = float(flow_data.get('IAT_mean', 0.0))
                iat_std = float(flow_data.get('IAT_std', 0.0))
            except KeyError:
                continue

            result = sf.stealth_signature_analysis(u_ports, sa_ratio, iat_mean, iat_std)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            if result != "Normal":
                if curr_attack_state != result:
                    curr_attack_state = result

                    os.makedirs("logs", exist_ok=True)
                    log_msg = f"[{timestamp}] Stealth Anomaly: {result} Started!\n"
                    with open("logs/ids_alerts.log", "a", encoding="utf-8") as log_file:
                        log_file.write(log_msg)

                    print(f"\n [!!!] ATTACK DETECTED: {result} | Ports: {u_ports}, S/A: {sa_ratio:.2f}, IAT: {iat_mean:.2f}s \n")

                else:
                    print(f" [{timestamp}] ... {result} is ongoing ...")

            else:
                if curr_attack_state != "Normal":
                    print(f"\n [+] Traffic normalized. Attack '{curr_attack_state}' has ended\n")
                    curr_attack_state = "Normal"
                else:
                    print(f" [{timestamp}] Stealth metrics stable (Ports: {u_ports}, S/A: {sa_ratio:.2f}) \n")

    except KeyboardInterrupt:
        print("\n Shutting down... \n")

    finally:
        consumer.close()

if __name__ == "__main__":
    Consumer_Stealth_func()