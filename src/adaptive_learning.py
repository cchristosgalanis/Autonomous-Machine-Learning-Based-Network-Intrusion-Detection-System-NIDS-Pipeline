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
    
    # ensure model directory exists
    os.makedirs("models/features", exist_ok=True)

    # Load Historical Data
    if not os.path.exists(historical_path):
        print(f"\n Critical Error: Historical data not found at {historical_path} \n")
        return
        
    print(f"\n Loading historical data from {historical_path}... \n")
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
            print(f"\n Data fusion complete. Added {len(X_adapt)} new samples \n")
        else:
            print("\n Adaptive dataset is empty. Proceeding with historical data only \n")
    else:
        print("\n Adaptive data not found. Proceeding with historical data only \n")

    print(f"\n Total dataset size before balancing: {len(X_combined)} samples \n")

    # --- dynamic data balancing ---
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

    print(f"\n Total dataset size for training after balancing: {len(X_combined)} samples. \n")

    #  Shuffle the combined dataset securely
    print("\n Shuffling data to prevent catastrophic forgetting... \n ")
    X_shuffled, y_shuffled = shuffle(X_combined, y_combined, random_state=42)

    # Feature Scaling
    print("\n Scaling features (StandardScaler)...\n ")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_shuffled)

    # Train the Neural Network
    print("\n Training Neural Network (MLPClassifier)...")
    #can adjust hyperparameters based on your original non_linear.py logic
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

    # save the updated model and scaler
    print(f"\n Saving updated model and scaler to disk...")
    joblib.dump(model, model_out_path)
    joblib.dump(scaler, scaler_out_path)
    
    print(f"\n Adaptive retraining complete. New weights deployed to {model_dir}/")

    # delete the temporary batch to free up space and prepare for the next cycle
    if os.path.exists(adaptive_path):
        try:
            os.remove(adaptive_path)
            print(f"\n Cleaned up temporary training batch: {adaptive_path}")
        except Exception as e:
            print(f"\n Failed to delete {adaptive_path}: {e}")

if __name__ == "__main__":
    perform_adaptive_retraining()