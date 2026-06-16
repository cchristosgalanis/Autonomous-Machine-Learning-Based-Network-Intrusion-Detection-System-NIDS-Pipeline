import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.dates as mdates
from sklearn import metrics

def create_charts(csv_file="system_metrics.csv", output_dir="charts"):
    if not os.path.exists(csv_file):
        print(f"=====\n")
        return
    os.makedirs(output_dir, exist_ok=True)

    
    df = pd.read_csv(csv_file)

    df['Datetime'] = pd.to_datetime(df['Datetime'])

    sns.set_theme(style="darkgrid")
    
    metrics = {
    "CPU_Usage_%": ("CPU Usage (%)", "#1f77b4", "cpu_usage.png"),
    "RAM_Usage_%": ("RAM Usage (%)", "#ff7f0e", "ram_usage.png"),
    "Disk_Write_MB/s": ("Disk Write Speed (MB/s)", "#2ca02c", "disk_io.png"),
    "GPU_Usage_%": ("GPU Usage (%)", "#d62728", "gpu_usage.png")
}

    print(f"Read datas {csv_file}...")

    for column, (title, color, filename) in metrics.items():
        if column not in df.columns or (column == "GPU_Usage_%" and df[column].max() == 0):
            continue

     
        plt.figure(figsize=(12, 6))

        
        sns.lineplot(data=df, x='Datetime', y=column, color=color, linewidth=2)

        
        plt.title(title, fontsize=16, fontweight='bold', pad=15)
        plt.xlabel("Χρονική Στιγμή", fontsize=12)
        plt.ylabel(title, fontsize=12)

        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)

        plt.tight_layout()

    
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300)
        

        plt.close()
        
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    create_charts()