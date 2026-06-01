import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import joblib

def perform_adaptive_retraining():
    print("\n --- [Adaptive Learning] Initializing Retraining Process ---")
    
    historical_path = "models/historical_attacks.npz" 
    adaptive_path = "models/features/adaptive_batch.npz"
    
    model_dir = "models"
    model_out_path = os.path.join(model_dir, "mlp_model.joblib")
    scaler_out_path = os.path.join(model_dir, "scaler.joblib")
    
    # Ensure model directory exists
    os.makedirs("models/features", exist_ok=True)

    # Load Historical Data
    if not os.path.exists(historical_path):
        print(f" [!] Critical Error: Historical data not found at {historical_path}")
        return
        
    print(f" [+] Loading historical data from {historical_path}...")
    historical_data = np.load(historical_path)
    X_hist = historical_data['X']
    y_hist = historical_data['y']
    
    X_combined = X_hist
    y_combined = y_hist
    
    # Load Adaptive Data (if available)
    if os.path.exists(adaptive_path):
        print(f" [+] Loading new adaptive data from {adaptive_path}...")
        adaptive_data = np.load(adaptive_path)
        X_adapt = adaptive_data['X']
        y_adapt = adaptive_data['y']
        
        if len(X_adapt) > 0:
            # Fusion
            X_combined = np.vstack((X_hist, X_adapt))
            y_combined = np.concatenate((y_hist, y_adapt))
            print(f" [+] Data fusion complete. Added {len(X_adapt)} new samples.")
        else:
            print(" [i] Adaptive dataset is empty. Proceeding with historical data only.")
    else:
        print(" [i] Adaptive data not found. Proceeding with historical data only.")

    print(f" [+] Total dataset size before balancing: {len(X_combined)} samples.")

    # --- DYNAMIC DATA BALANCING ---
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
    # --------------------------------

    print(f" [+] Total dataset size for training after balancing: {len(X_combined)} samples.")

    # 3. Shuffle the combined dataset securely
    print(" [+] Shuffling data to prevent catastrophic forgetting...")
    X_shuffled, y_shuffled = shuffle(X_combined, y_combined, random_state=42)

    # 4. Feature Scaling
    print(" [+] Scaling features (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_shuffled)

    # 5. Train the Neural Network
    print(" [+] Training Neural Network (MLPClassifier)...")
    # You can adjust hyperparameters based on your original non_linear.py logic
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
    print(f" [+] Training completed. Final model score (accuracy): {model.score(X_scaled, y_shuffled):.4f}")

    # 6. Save the updated Model and Scaler
    print(" [+] Saving updated model and scaler to disk...")
    joblib.dump(model, model_out_path)
    joblib.dump(scaler, scaler_out_path)
    
    print(f" [SUCCESS] Adaptive retraining complete. New weights deployed to {model_dir}/")

if __name__ == "__main__":
    perform_adaptive_retraining()