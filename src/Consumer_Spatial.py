from confluent_kafka import Consumer
import json
import numpy as np

def Spatial_Consumer():
    # kafka settings
    conf = {
        'bootstrap.servers': 'localhost:9092',
        'group.id': 'spatial_minhash_group',
        'auto.offset.reset': 'latest', # real-time processing: start from latest messages
        'enable.auto.commit': True
    }


    consumer = Consumer(conf)
    spatial_topic = 'spatial-minhash-signatures'
    consumer.subscribe([spatial_topic])

    print(f"\n[Spatial Consumer] Listening to '{spatial_topic}' for Botnet signatures...\n")

    #store previous signatures
    previous_signatures = {}
    
    # jaccard threshold
    JACCARD_THRESHOLD = 0.70  

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue
            if msg.error():
                print(f"Consumer Error: {msg.error()}")
                continue

            # decode JSON
            payload = json.loads(msg.value().decode('utf-8'))
            
            target_port = payload['target_port']
            k_size = payload['k_size']
            
            # list -> numpy array
            current_sig = np.array(payload['signature'])

            if np.all(current_sig == -1):
                previous_signatures[target_port] = current_sig
                continue

            # compute Jaccard similarity
            if target_port in previous_signatures:
                prev_sig = previous_signatures[target_port]
                
                # check if previous wa empty
                if not np.all(prev_sig == -1):
                    # vectorized comparison of signatures
                    matches = np.sum(current_sig == prev_sig)
                    
                    # jaccard score based on matches and k_size
                    jaccard_score = matches / k_size
                    
                    print(f"Port {target_port} | MinHash Matches: {matches}/{k_size} | Jaccard: {jaccard_score:.3f}")

                    # check
                    if jaccard_score > JACCARD_THRESHOLD:
                        print(f"\n ALERT: DISTRIBUTED LOW-RATE FLOOD DETECTED!")
                        print(f"\n Spatial Correlation exceeded threshold (Score: {jaccard_score:.3f})")
                        print(f"\n A coordinated Botnet is targeting Port {target_port} \n")

            # update the memory for the next time window
            previous_signatures[target_port] = current_sig

    except KeyboardInterrupt:
        print("\nStopping Spatial Consumer...")
    finally:
        consumer.close()

if __name__ == "__main__":
    Spatial_Consumer()