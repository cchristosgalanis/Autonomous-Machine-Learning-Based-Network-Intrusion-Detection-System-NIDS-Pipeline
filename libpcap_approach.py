import pcapy
import pandas as pd
import time 
import numpy as np
import socket
from struct import unpack

def libpcap_capture(capture_duration, interface):
    print(f"\n [Libpcap] Starting high-speed capture on {interface}...")
    
    # Open interface: max bytes, promiscuous, timeout
    cap = pcapy.open_live(interface, 65536, 1, 100)
    cap.setfilter("tcp or udp") # BPF filter at kernel level

    flow_hash_map = {}
    start_time = time.time()

    while (time.time() - start_time) < capture_duration:
        try:
            # Capture packet (header contains timestamp and length)
            header, data = cap.next()
            if not data:
                continue

            # Ethernet header is 14 bytes
            # IP header starts at byte 14
            ip_header = data[14:34]
            iph = unpack('!BBHHHBBH4s4s', ip_header)
            
            # Extract basic IP info
            version_ihl = iph[0]
            ihl = (version_ihl & 0xF) * 4 # IP Header Length
            protocol = iph[6]
            src_ip = socket.inet_ntoa(iph[8])
            dst_ip = socket.inet_ntoa(iph[9])

            # Transport layer (TCP/UDP) starts after IP header
            trans_header = data[14 + ihl : 14 + ihl + 4]
            src_port, dst_port = unpack('!HH', trans_header)

            ips = tuple(sorted([src_ip, dst_ip]))
            ports = tuple(sorted([src_port, dst_port]))
            flow_key = (ips[0], ips[1], ports[0], ports[1], protocol)

            if flow_key not in flow_hash_map:
                flow_hash_map[flow_key] = {'packets': 0, 'bytes': 0}
            
            flow_hash_map[flow_key]['packets'] += 1
            flow_hash_map[flow_key]['bytes'] += len(data)

        except Exception:
            continue

    # --- Processing results into DataFrame ---
    flow_data = []
    for key, metrics in flow_hash_map.items():
        pkts = metrics['packets']
        byts = metrics['bytes']
        
        flow_data.append({
            'flow_byts_s': byts / capture_duration,
            'flow_pkts_s': pkts / capture_duration,
            'pkt_len_mean': byts / pkts
        })

    return pd.DataFrame(flow_data)

