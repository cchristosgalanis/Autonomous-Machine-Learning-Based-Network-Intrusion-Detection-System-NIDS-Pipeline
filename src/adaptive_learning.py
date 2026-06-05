import os
import numpy as np
import psycopg2
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import joblib

def fetch_adaptive_data_from_db():
    """
    get datas from the database that are either very confidently labeled as Normal 
    (probability <= T_SAFE), or very confidently labeled as Attack (probability >= CONFIDENCE_THRESHOLD),
    or come from the Host IP (which we label as Benign).
    This function is used in the adaptive learning process to fetch new data for retraining.
    """
    HOST_IP = os.getenv('HOST_IP', '127.0.0.1')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASS = os.getenv('DB_PASS', 'yourpassword')
    DB_NAME = os.getenv('DB_NAME', 'postgres')

    T_SAFE = 0.15
    CONFIDENCE_THRESHOLD = 0.85

    X_adapt = []
    y_adapt = []

    try:
        conn = psycopg2.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, dbname=DB_NAME)
        cursor = conn.cursor()

        # SQL Query: get only recent records (last 7 days) with high confidence or Host  traffic
        query = f"""
            SELECT source_ip, velocity, acceleration, iat_mean, sa_ratio, jaccard_score, entropy_shannon, prediction
            FROM network_traffic_events
            WHERE time > NOW() - INTERVAL '7 days'
              AND (probability <= {T_SAFE} OR probability >= {CONFIDENCE_THRESHOLD} OR source_ip = '{HOST_IP}');
        """
        cursor.execute(query)
        records = cursor.fetchall()

        for row in records:
            ip = row[0]
            features = list(row[1:7])
            label = 0 if ip == HOST_IP else int(row[7])
            X_adapt.append(features)
            y_adapt.append(label)

        cursor.close()
        conn.close()

        return np.array(X_adapt), np.array(y_adapt)
    except Exception as e:
        print(f"\n  DB Fetch Error in Adaptive Learning: {e}")
        return np.array([]), np.array([])


def perform_adaptive_retraining():
    print("\n --- [Adaptive Learning] Initializing Retraining Process ---")
    
    historical_path = "models/historical_attacks.npz" 
    model_dir = "models"
    model_out_path = os.path.join(model_dir, "mlp_model.joblib")
    scaler_out_path = os.path.join(model_dir, "scaler.joblib")

    # get historical data for attacks
    if not os.path.exists(historical_path):
        print(f"\n Critical Error: Historical data not found at {historical_path} \n")
        return
        
    print(f"\n Loading historical data from {historical_path}... \n")
    historical_data = np.load(historical_path)
    X_combined = historical_data['X']
    y_combined = historical_data['y']
    
    # load new data from the database that has been sanitized and labeled by the sanitization service
    print(" [+] Fetching accumulated adaptive data from Database (Last 7 Days)...")
    X_adapt, y_adapt = fetch_adaptive_data_from_db()
    
    if len(X_adapt) > 0:
        # data fusion
        X_combined = np.vstack((X_combined, X_adapt))
        y_combined = np.concatenate((y_combined, y_adapt))
        print(f"\n Data fusion complete. Added {len(X_adapt)} new verified samples from DB. \n")
    else:
        print("\n No new confident data found in DB. Proceeding with historical data only. \n")

    print(f"\n Total dataset size before balancing: {len(X_combined)} samples \n")

    # dynamic balancing
    idx_benign = np.where(y_combined == 0)[0]
    idx_attack = np.where(y_combined == 1)[0]

    attacks_to_keep = max(len(idx_benign), 5000)
    
    if len(idx_attack) > attacks_to_keep:
        print(f" [+] Undersampling attacks from {len(idx_attack)} down to {attacks_to_keep} to match benign traffic...")
        np.random.shuffle(idx_attack)
        idx_attack = idx_attack[:attacks_to_keep]

    balanced_indices = np.concatenate((idx_benign, idx_attack))
    X_combined = X_combined[balanced_indices]
    y_combined = y_combined[balanced_indices]

    print(f"\n Total dataset size for training after balancing: {len(X_combined)} samples. \n")

    # shuffle and scaling 
    print("\n Shuffling data to prevent catastrophic forgetting... \n ")
    X_shuffled, y_shuffled = shuffle(X_combined, y_combined, random_state=42)

    print("\n Scaling features (StandardScaler)...\n ")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_shuffled)

    # retrain neural network
    print("\n Training Neural Network (MLPClassifier)...")
    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    model.fit(X_scaled, y_shuffled)
    print(f"\n Training completed. Final model score (accuracy): {model.score(X_scaled, y_shuffled):.4f}")

    # save new model's weights
    print(f"\n Saving updated model and scaler to disk...")
    joblib.dump(model, model_out_path)
    joblib.dump(scaler, scaler_out_path)
    
    print(f"\n Adaptive retraining complete. New weights deployed to {model_dir}/")

if __name__ == "__main__":
    perform_adaptive_retraining()