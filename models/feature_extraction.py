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

def compute_features(raw_csv_path, label_value):
    """
    read raw flows, simulate sliding windows, and extract the 6-dimensional feature vector
    """
    print(f"\n [+] Processing: {raw_csv_path} ...")
    df_raw = pd.read_csv(raw_csv_path)

    model, scaler = nl.load_model_and_scaler()
    if model is None:
        raise Exception("Error: Model files not found in models/ directory.")
    
    w_train = 20
    buffer = deque(maxlen=w_train)
    residual_memory = deque(maxlen=3)
    X_dataset = []

    for index, row in df_raw.iterrows():
        # volumetric extraction with safe gets
        t_vol = float(row.get('flow_byts_s', row.get('T_vol', 0.0)))
        n_req = float(row.get('flow_pkts_s', row.get('N_req', 0.0)))
        s_len = float(row.get('pkt_len_mean', row.get('S_len', 0.0)))

        buffer.append(np.array([t_vol, n_req, s_len]))

        velocity, acceleration = 0.0, 0.0
        if len(buffer) >= w_train:
            analysis_batch = np.array(list(buffer))[-w_train:]
            entropy_res = nl.entropy_based_stab_check(analysis_batch[:, 0], analysis_batch[:, 1], analysis_batch[:, 2], w_train, model, scaler)
            residual_memory.append(float(np.mean(entropy_res)))

            if len(residual_memory) >= 2:
                velocity, acceleration = nl.compute_kinematic(list(residual_memory))

        # stealth & spatial features
        iat_mean = float(row.get('iat_mean', row.get('IAT_mean', 0.0)))
        sa_ratio = float(row.get('syn_ack_ratio', row.get('SA_ratio', 0.0)))
        jaccard_score = float(row.get('Jaccard_score', row.get('jaccard_score', 0.0)))
        
        domain = str(row.get('Domain', row.get('domain', '')))
        entropy_shannon = nl.shannon_entropy(domain) if (domain and domain.lower() != 'nan') else 0.0

        if len(residual_memory) >= 2:
            X_dataset.append([abs(velocity), abs(acceleration), iat_mean, sa_ratio, jaccard_score, entropy_shannon])

    X_np = np.array(X_dataset)
    y_np = np.full(shape=(X_np.shape[0],), fill_value=label_value, dtype=int)
    return X_np, y_np

if __name__ == "__main__":
    # benign files 
    benign_files = [
            "capture.csv", 
            "benign.csv",
            "normal_curl_01.csv",
            "normal_idle_01.csv",
            "normal_ping_01.csv"
    ] 
    
    
    # attack files
    attack_files = [
        "flows.csv", 
        "slowdos.csv", 
        "syn_flood.csv", 
        "tcp_scan.csv", 
        "udp_flood.csv", 
        "udp_scan.csv"
    ]

    print("\n --- Starting Offline Feature Extraction ---")
    
    all_X = []
    all_y = []

    # 1. Processing Benign
    print("\n[+] Processing Benign Traffic...")
    for f in benign_files:
        if os.path.exists(f):
            X, y = compute_features(f, label_value=0)
            all_X.append(X)
            all_y.append(y)
        else:
            print(f" [!] Warning: File {f} not found, skipping.")

    # 2. Processing Attacks
    print("\n[+] Processing Attack Traffic...")
    for f in attack_files:
        if os.path.exists(f):
            X, y = compute_features(f, label_value=1)
            all_X.append(X)
            all_y.append(y)
        else:
            print(f" [!] Warning: File {f} not found, skipping.")

    # 3. Final Fusion
    if all_X:
        X_final = np.vstack(all_X)
        y_final = np.concatenate(all_y)

        # Δημιουργία φακέλου features και αποθήκευση
        os.makedirs("features", exist_ok=True)
        np.savez_compressed("features/final_features.npz", X=X_final, y=y_final)
        
        print(f"\n [SUCCESS] Final dataset created: features/final_features.npz")
        print(f" Total samples: {X_final.shape[0]}")
        print(f" Benign samples: {np.sum(y_final == 0)}")
        print(f" Attack samples: {np.sum(y_final == 1)}")
    else:
        print("\n [!] Error: No data processed. Check if files exist.")