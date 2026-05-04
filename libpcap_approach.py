import pcapy
import pandas as pd
import time 

def libpcap_capture(capture_duration, interface):
    print(f"\n [Libpcap] Starting time-window capture on {interface}...")
    
    # Open interface: max bytes, promiscuous, timeout
    cap = pcapy.open_live(interface, 65536, 1, 100)
    cap.setfilter("ip") # BPF filter at kernel level

    start_time = time.time()

    # counters
    total_packets = 0
    total_bytes = 0

    while (time.time() - start_time) < capture_duration:
        try:
            # Capture packet
            header, data = cap.next()
            if not data:
                continue
            
            total_packets += 1
            total_bytes += len(data)

        except Exception:
            continue

    #comppute metrics
    
    flow_byts_s = total_bytes / capture_duration
    flow_pkts_s = total_packets / capture_duration
    pkt_len_mean = (total_bytes / total_packets) if total_packets > 0 else 0

    # return a dataframe
    flow_data = [{
        'flow_byts_s': flow_byts_s,
        'flow_pkts_s': flow_pkts_s,
        'pkt_len_mean': pkt_len_mean
    }]

    return pd.DataFrame(flow_data)