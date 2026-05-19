from confluent_kafka import Consumer
import json
import time
import os 
import stateful_functions as sf
import csv # Προσθήκη για καταγραφή στο CSV

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

    print(f"\n Stealth Analyzer connected to '{topic_name}'. Waiting for data ...\n")
    print("\n Stateful/Stealth Analysis initialized \n")

    # --- ΠΡΟΣΘΗΚΗ: Αρχικοποίηση αρχείου CSV για τα γραφήματα ---
    csv_file_path = "logs/stealth_metrics.csv"
    os.makedirs("logs", exist_ok=True)
    
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Πρόσθεσα και τα U_ports και τον τύπο επίθεσης (Attack_Type) για καλύτερα γραφήματα!
            writer.writerow(["Timestamp", "IAT_mean", "IAT_std", "SA_ratio", "U_ports", "Is_Attack", "Attack_Type"])
    # -------------------------------------------------------------

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

            # Κλήση της δικής σου συνάρτησης
            result = sf.stealth_signature_analysis(u_ports, sa_ratio, iat_mean, iat_std)

            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            current_plot_time = time.time() # Ακριβής χρόνος για τον άξονα Χ του διαγράμματος

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

            # --- ΠΡΟΣΘΗΚΗ: ΣΥΝΕΧΗΣ ΚΑΤΑΓΡΑΦΗ ΔΕΔΟΜΕΝΩΝ ΣΤΟ CSV ---
            # Αν το result δεν είναι "Normal", τότε έχουμε επίθεση (1), αλλιώς (0)
            current_alert_status = 1 if result != "Normal" else 0
            
            with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([current_plot_time, iat_mean, iat_std, sa_ratio, u_ports, current_alert_status, result])
            # ------------------------------------------------------

    except KeyboardInterrupt:
        print("\n Shutting down... \n")

    finally:
        consumer.close()

if __name__ == "__main__":
    Consumer_Stealth_func()