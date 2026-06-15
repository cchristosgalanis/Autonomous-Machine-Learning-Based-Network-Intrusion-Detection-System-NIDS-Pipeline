import pandas as pd
import numpy as np
from collections import deque
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

import non_linear as nl

def compute_features_offline(raw_csv_path, label_value):
    """
    Read raw historical flows, simulate the sliding window, 
    and extract the 6-dimensional feature vector.
    """
    print(f"\n [+] Processing historical data: {raw_csv_path} ...")

    model, scaler = nl.load_model_and_scaler()
    if model is None:
        raise Exception("Error: Model files not found in models/ directory.")
    
    w_train = 20
    buffer = deque(maxlen=w_train)
    residual_memory = deque(maxlen=3)
    X_dataset = []

    # Full file load into memory (safe since .gz is small)
    df_raw = pd.read_csv(raw_csv_path)

    for index, row in df_raw.iterrows():
        # volumetric extraction with safe gets
        t_vol = float(row.get('Flow Byts/s', row.get('flow_byts_s', row.get('T_vol', 0.0))))
        n_req = float(row.get('Flow Pkts/s', row.get('flow_pkts_s', row.get('N_req', 0.0))))
        s_len = float(row.get('Pkt Len Mean', row.get('pkt_len_mean', row.get('S_len', 0.0))))

        buffer.append(np.array([t_vol, n_req, s_len]))

        velocity, acceleration = 0.0, 0.0
        if len(buffer) >= w_train:
            analysis_batch = np.array(list(buffer))[-w_train:]
            entropy_res = nl.entropy_based_stab_check(
                analysis_batch[:, 0], analysis_batch[:, 1], analysis_batch[:, 2], 
                w_train, model, scaler
            )
            residual_memory.append(float(np.mean(entropy_res)))

            if len(residual_memory) >= 2:
                velocity, acceleration = nl.compute_kinematic(list(residual_memory))

        # stealth & spatial features
        iat_mean = float(row.get('Flow IAT Mean', row.get('iat_mean', row.get('IAT_mean', 0.0))))
        
        syn_ack_ratio = row.get('syn_ack_ratio', row.get('SA_ratio'))
        if syn_ack_ratio is not None:
            sa_ratio = float(syn_ack_ratio)
        else:
            syn = float(row.get('SYN Flag Cnt', 0.0))
            ack = float(row.get('ACK Flag Cnt', 0.0))
            if ack > 0:
                sa_ratio = syn / ack
            elif syn > 0:
                sa_ratio = 999.0
            else:
                sa_ratio = 0.0

        jaccard_score = float(row.get('Jaccard_score', row.get('jaccard_score', 0.0)))
        
        domain = str(row.get('Domain', row.get('domain', '')))
        entropy_shannon = nl.shannon_entropy(domain) if (domain and domain.lower() != 'nan') else 0.0

        if len(residual_memory) >= 2:
            X_dataset.append([abs(velocity), abs(acceleration), iat_mean, sa_ratio, jaccard_score, entropy_shannon])

    X_np = np.array(X_dataset)
    y_np = np.full(shape=(X_np.shape[0],), fill_value=label_value, dtype=int)
    return X_np, y_np

if __name__ == "__main__":
    
    # Try multiple common paths for the raw gzipped CSV
    attack_file = "topical_src/merged_attacks.csv.gz"
    if not os.path.exists(attack_file):
        attack_file = "models/merged_attacks.csv.gz"
    if not os.path.exists(attack_file):
        attack_file = "merged_attacks.csv.gz"
        
    output_file = "models/historical_attacks.npz"

    print("\n --- Starting ONE-TIME Historical Feature Extraction ---")
    
    if os.path.exists(attack_file):
        X_attack, y_attack = compute_features_offline(attack_file, label_value=1)
        
        # Save directly to .npz
        np.savez_compressed(output_file, X=X_attack, y=y_attack)
        
        print(f"\n [SUCCESS] Historical attacks dataset created: {output_file}")
        print(f" Total attack samples extracted: {X_attack.shape[0]}")
    else:
        print(f"\n [!] Error: File {attack_file} not found. Cannot proceed.")