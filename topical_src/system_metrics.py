import psutil
import time
import csv
import os
from datetime import datetime

try:
    import GPUtil
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

def monitor_system(csv_filename="system_metrics.csv", interval=1.0):
    file_exists = os.path.isfile(csv_filename)
    
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow(["Timestamp", "Datetime", "CPU_Usage_%", "RAM_Usage_%", "Disk_Write_MB/s", "GPU_Usage_%"])
        
        last_disk_io = psutil.disk_io_counters()
        last_write_bytes = last_disk_io.write_bytes if last_disk_io else 0
        
        try:
            while True:
                time.sleep(interval)
                
                # time
                current_time = time.time()
                dt_string = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S")
                
                # cpu & ram
                cpu_perc = psutil.cpu_percent(interval=None)
                ram_perc = psutil.virtual_memory().percent
                
                # ssd
                current_disk_io = psutil.disk_io_counters()
                current_write_bytes = current_disk_io.write_bytes if current_disk_io else 0
                
                bytes_written = current_write_bytes - last_write_bytes
                write_mb_per_sec = (bytes_written / interval) / (1024 * 1024)
                last_write_bytes = current_write_bytes
                
                # gpu
                gpu_perc = 0.0
                if HAS_GPU:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu_perc = gpus[0].load * 100
                        
                # csv
                writer.writerow([
                    current_time, 
                    dt_string, 
                    cpu_perc, 
                    ram_perc, 
                    round(write_mb_per_sec, 4), 
                    round(gpu_perc, 2)
                ])
                
                file.flush()
                
        except KeyboardInterrupt as e:
            print(e)

if __name__ == "__main__":
    monitor_system(interval=1.0)