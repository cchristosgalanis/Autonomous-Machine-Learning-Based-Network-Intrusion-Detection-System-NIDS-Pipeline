import json
import numpy as np
import ipaddress
import os

def is_valid_ip(ip):
    try:
        ipaddress.ip_address(ip)
        return True
    except ValueError:
        return False
    
def sanitization_logic(training_candidates_path):
    """
    Sanitizes raw training data based on confidence thresholds and IP validity.
    - Filters out invalid IPs.
    - Labels Host traffic as Benign (0).
    - Labels high-confidence attacks as Attack (1).
    - Filters out 'uncertain' samples from the training batch.
    """

    HOST_IP = os.getenv('HOST_IP', '127.0.0.1')
    T_SAFE = 0.05
    CONFIDENCE_THRESHOLD = 0.95
    
    sanitized_X = []
    sanitized_y = []

    if not os.path.exists(training_candidates_path):
        print(f" [!] File not found: {training_candidates_path}")
        return None, None
    
    with open(training_candidates_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                entity_id = data.get("entity_id")
                prob = data.get("probability", 0.0)
                
                # filter: ensure entity is a valid IP
                if not is_valid_ip(entity_id):
                    continue
                
                # filtering strategy
                # We only want samples that we are highly confident about
                is_confident_attack = (prob >= CONFIDENCE_THRESHOLD)
                is_confident_benign = (prob <= T_SAFE)
                
                # labeling logic
                # If it's the Host, force label as Benign (0)
                if entity_id == HOST_IP:
                    sanitized_X.append(data["features"])
                    sanitized_y.append(0)
                
                elif is_confident_benign: 
                    sanitized_X.append(data["features"])
                    sanitized_y.append(0)
                
                # if it's a confident attack, keep it as Attack (1)
                elif is_confident_attack:
                    sanitized_X.append(data["features"])
                    sanitized_y.append(1)
                
                # ignore 'is_uncertain' samples during training to prevent poisoning
                else:
                    continue

            except Exception as e:
                continue

    return np.array(sanitized_X), np.array(sanitized_y)

if __name__ == "__main__":
    X, y = sanitization_logic("logs/training_candidates.jsonl")
    
    if X is not None and len(X) > 0:
        np.savez("features/adaptive_batch.npz", X=X, y=y)
        print(f"[*] Sanitization complete. {len(X)} samples ready for training.")