import pcapy
import pandas as pd
import time 
import dpkt
import numpy as np

def libpcap_capture(capture_duration, interface,flow_states):
    print(f"\n [Libpcap] Starting time-window capture on {interface}...")
    
    # Open interface: max bytes, promiscuous, timeout
    cap = pcapy.open_live(interface, 65536, 1, 100)
    cap.setfilter("ip") # BPF filter at kernel level

    start_time = time.time()

    # volumetric variables
    total_packets = 0
    total_bytes = 0

    # stealth variables
    unique_dst_port = set()
    syn_count = 0
    ack_count = 0

    curr_window_iats = []

    while (time.time() - start_time) < capture_duration:
        try:
            # Capture packet
            header, data = cap.next()
            if not data:
                continue
            
            total_packets += 1
            total_bytes += len(data)

            #decode for stealth attack
            eth = dpkt.ethernet.Ethernet(data)

            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data

                if isinstance(ip.data, dpkt.ip.IP):
                    tcp = ip.data

                    # unique destination ports
                    unique_dst_port.add(tcp.dport)

                    if tcp.flags & dpkt.tcp.TH_SYN:
                        syn_count += 1
                    if tcp.flags & dpkt.tcp.TH_ACK:
                        ack_count += 1

                    #create a key for a set
                    flow_key = (ip.src, tcp.sport, ip.dst, tcp.dport)
                    packet_time = time.time()

                    if flow_key in flow_states:
                        iat = packet_time - flow_states[flow_key]
                        curr_window_iats.append(iat)


        except Exception:
            continue

# ----- comppute metrics -------
    
    # volumetric metrics
    flow_byts_s = total_bytes / capture_duration
    flow_pkts_s = total_packets / capture_duration
    pkt_len_mean = (total_bytes / total_packets) if total_packets > 0 else 0

    #stealth metrics
    port_scan_intensity = len(unique_dst_port)
    syn_ack_ratio = syn_count / ack_count if ack_count > 0 else syn_count

    # iat mean and iat std
    iat_mean = float(np.mean(curr_window_iats)) if curr_window_iats else 0.0
    iat_std = float(np.std(curr_window_iats)) if len(curr_window_iats) > 1 else 0.0

    # return a dataframe
    flow_data = [{
        'flow_byts_s': flow_byts_s,
        'flow_pkts_s': flow_pkts_s,
        'pkt_len_mean': pkt_len_mean,
        'unique_ports': port_scan_intensity,
        'syn_ack_ratio': float(syn_ack_ratio),
        'iat_mean': iat_mean,
        'iat-std': iat_std
    }]

    return pd.DataFrame(flow_data)