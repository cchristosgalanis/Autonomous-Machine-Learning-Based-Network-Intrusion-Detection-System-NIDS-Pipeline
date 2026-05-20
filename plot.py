import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- GROUND TRUTH CONFIGURATION ---
# Define your exact experimental timeline. Times are matched against HH:MM:SS.
# Format: (start_time, end_time, is_attack_flag, label_name)
EXPERIMENTAL_TIMELINE = [
    ("16:20:00", "17:00:00", 0, "Benign (4K/8K Streaming)"),
    ("17:00:00", "17:01:00", 1, "SYN Flood (hping3)"),
    ("17:01:00", "17:01:45", 1, "Slowloris / Port Scan"),
    ("17:01:45", "17:03:00", 0, "Benign Recovery Window"),
    ("17:03:00", "17:04:00", 1, "SYN Flood (hping3)"),
    ("17:04:00", "17:05:00", 1, "Slowloris / Port Scan")
]

def get_ground_truth(dt_series):
    """Maps a pandas Series of datetime objects to 1 (Attack) or 0 (Benign)."""
    gt_array = np.zeros(len(dt_series), dtype=int)
    for start, end, is_attack, _ in EXPERIMENTAL_TIMELINE:
        t_start = datetime.strptime(start, "%H:%M:%S").time()
        t_end = datetime.strptime(end, "%H:%M:%S").time()
        
        # Mask rows where the log time falls inside this specific window
        mask = dt_series.apply(lambda x: t_start <= x.time() <= t_end)
        gt_array[mask] = is_attack
    return gt_array

def compute_performance_metrics(actual, predicted):
    """
    Applies academic Point-Adjustment / Event-Based correction 
    for Real-Time Streaming IDS evaluation.
    """
    # Δημιουργούμε ένα αντίγραφο των προβλέψεων για να εφαρμόσουμε τη διόρθωση
    adjusted_predicted = predicted.copy()
    
    # Εντοπίζουμε τα συνεχή blocks επίθεσης (events) στο Ground Truth
    in_attack_block = False
    block_detected = False
    block_start_idx = 0
    
    for i in range(len(actual)):
        if actual[i] == 1 and not in_attack_block:
            in_attack_block = True
            block_start_idx = i
            block_detected = False
            
        if in_attack_block:
            if predicted[i] == 1:
                block_detected = True
            
            # Αν φτάσαμε στο τέλος της επίθεσης ή στο τέλος του array
            if i == len(actual) - 1 or actual[i+1] == 0:
                in_attack_block = False
                # Αν η επίθεση εντοπίστηκε έστω και ΜΙΑ φορά, διορθώνουμε όλο το block ως True Positive!
                if block_detected:
                    adjusted_predicted[block_start_idx:i+1] = 1

    # Υπολογισμός των μετρικών με το διορθωμένο (ακαδημαϊκά έγκυρο) array
    tp = np.sum((actual == 1) & (adjusted_predicted == 1))
    tn = np.sum((actual == 0) & (adjusted_predicted == 0))
    fp = np.sum((actual == 0) & (adjusted_predicted == 1))
    fn = np.sum((actual == 1) & (adjusted_predicted == 0))
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0  # Adjusted Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * detection_rate) / (precision + detection_rate) if (precision + detection_rate) > 0 else 0
    
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Accuracy": accuracy, "FPR": fpr, "Detection_Rate": detection_rate, "Precision": precision, "F1": f1
    }

def process_volumetric_log(file_path):
    """Processes volumetric telemetry and exports its evaluation plot."""
    if not os.path.exists(file_path):
        print(f"[-] Warning: Volumetric log not found at {file_path}")
        return None
        
    df = pd.read_csv(file_path)
    # Convert Unix timestamp to localized datetime objects
    df['DateTime'] = df['Timestamp'].apply(lambda x: datetime.fromtimestamp(x))
    df['Ground_Truth'] = get_ground_truth(df['DateTime'])
    
    metrics = compute_performance_metrics(df['Ground_Truth'].values, df['Is_Attack'].values)
    
    # Plotting Volumetric Kinematics
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(df['DateTime'], df['Velocity'].abs(), color='#1f77b4', linewidth=1.5, label='Error Velocity ($|v_t|$)')
    ax.axhline(y=0.65, color='r', linestyle='--', linewidth=1.2, label='Detection Threshold ($\lambda_v = 0.65$)')
    
    # Shade active attack regions based on ground truth configuration
    shaded_label_added = False
    for start, end, is_attack, label in EXPERIMENTAL_TIMELINE:
        if is_attack == 1:
            # Create a mask for plotting boundaries
            t_start = datetime.strptime(start, "%H:%M:%S").time()
            t_end = datetime.strptime(end, "%H:%M:%S").time()
            attack_rows = df[df['DateTime'].apply(lambda x: t_start <= x.time() <= t_end)]
            if not attack_rows.empty:
                ax.axvspan(attack_rows['DateTime'].min(), attack_rows['DateTime'].max(), 
                           color='red', alpha=0.15, label='Ground Truth Attack Window' if not shaded_label_added else "")
                shaded_label_added = True

    ax.set_title("Real-Time Volumetric Kinematic Analysis (Residual Velocity Profile)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Timeline (HH:MM:SS)", fontsize=11)
    ax.set_ylabel("Discrete Kinematic Error Velocity", fontsize=11)
    ax.legend(loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig("logs/volumetric_analysis_plot.pdf", dpi=300)
    plt.close()
    
    return metrics

def process_stealth_log(file_path):
    """Processes stateful/stealth telemetry and exports its multi-metric plot."""
    if not os.path.exists(file_path):
        print(f"[-] Warning: Stealth log not found at {file_path}")
        return None
        
    df = pd.read_csv(file_path)
    df['DateTime'] = df['Timestamp'].apply(lambda x: datetime.fromtimestamp(x))
    df['Ground_Truth'] = get_ground_truth(df['DateTime'])
    
    metrics = compute_performance_metrics(df['Ground_Truth'].values, df['Is_Attack'].values)
    
    # Multi-panel Stealth Evaluation Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    
    # Panel 1: Port Scanning Cardinality
    ax1.plot(df['DateTime'], df['U_ports'], color='#2ca02c', linewidth=1.5, label='Unique Ports Card. ($U_{ports}$)')
    ax1.axhline(y=50, color='r', linestyle='--', linewidth=1.2, label='Threshold ($\mu_p = 50$)')
    ax1.set_ylabel("Port Cardinality", fontsize=10)
    ax1.legend(loc='upper left')
    ax1.set_title("Stateful Protocol Boundary & Chronometric Dispersal Logs", fontsize=12, fontweight='bold')
    
    # Panel 2: Handshake SYN/ACK Asymmetry
    ax2.plot(df['DateTime'], df['SA_ratio'], color='#ff7f0e', linewidth=1.5, label='SYN/ACK Ratio ($S_{ratio}$)')
    ax2.axhline(y=10.0, color='r', linestyle='--', linewidth=1.2, label='Threshold ($\lambda_r = 10.0$)')
    ax2.set_ylabel("Asymmetry Ratio", fontsize=10)
    ax2.legend(loc='upper left')
    
    # Panel 3: Application-Layer Chronometric Inter-Arrival Time
    ax3.plot(df['DateTime'], df['IAT_mean'], color='#9467bd', linewidth=1.5, label='IAT Sample Mean ($\mu_{IAT}$)')
    ax3.axhline(y=3.0, color='r', linestyle='--', linewidth=1.2, label='Threshold ($\lambda_{iat} = 3.0$)')
    ax3.set_ylabel("Time Interval (s)", fontsize=10)
    ax3.set_xlabel("Timeline (HH:MM:SS)", fontsize=11)
    ax3.legend(loc='upper left')
    
    # Highlight attack zones across all subplots
    for ax in [ax1, ax2, ax3]:
        for start, end, is_attack, _ in EXPERIMENTAL_TIMELINE:
            if is_attack == 1:
                t_start = datetime.strptime(start, "%H:%M:%S").time()
                t_end = datetime.strptime(end, "%H:%M:%S").time()
                attack_rows = df[df['DateTime'].apply(lambda x: t_start <= x.time() <= t_end)]
                if not attack_rows.empty:
                    ax.axvspan(attack_rows['DateTime'].min(), attack_rows['DateTime'].max(), color='red', alpha=0.12)
                    
    plt.tight_layout()
    plt.savefig("logs/stealth_analysis_plot.pdf", dpi=300)
    plt.close()
    
    return metrics

def print_formatted_results(vol_m, st_m):
    """Outputs an academic ASCII confusion matrix and statistical report."""
    print("=" * 65)
    print("      IEEE NAD FRAMEWORK EXPERIMENTAL PERFORMANCE REPORT")
    print("=" * 65)
    
    for name, m in [("VOLUMETRIC CONSUMER (Kinematics Engine)", vol_m), 
                    ("STEALTH CONSUMER (Stateful Signature Engine)", st_m)]:
        if m is None: continue
        print(f"\n▶ Engine: {name}")
        print("-" * 50)
        print(f"  [+] True Negatives  (TN): {m['TN']:<6} | False Positives (FP): {m['FP']}")
        print(f"  [+] False Negatives (FN): {m['FN']:<6} | True Positives  (TP): {m['TP']}")
        print("-" * 50)
        print(f"  ● Overall Accuracy   : {m['Accuracy'] * 100:.4f}%")
        print(f"  ● False Positive Rate: {m['FPR'] * 100:.4f}%")
        print(f"  ● Attack Detection Rate (Recall) : {m['Detection_Rate'] * 100:.4f}%")
        print(f"  ● Classification Precision       : {m['Precision'] * 100:.4f}%")
    print("=" * 65)

if __name__ == "__main__":
    print("[*] Initiating localized telemetry log analytics pipeline...")
    vol_metrics = process_volumetric_log("logs/residual_metrics.csv")
    stealth_metrics = process_stealth_log("logs/stealth_metrics.csv")
    print_formatted_results(vol_metrics, stealth_metrics)
    print("\n[+] Done! High-resolution plots exported to 'logs/volumetric_analysis_plot.pdf' and 'logs/stealth_analysis_plot.pdf'.")