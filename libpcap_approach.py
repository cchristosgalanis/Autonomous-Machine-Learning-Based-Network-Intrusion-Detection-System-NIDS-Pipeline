import pcapy
import pandas as pd
import time 
import dpkt
import numpy as np
import socket

def libpcap_capture(capture_duration, interface, flow_states):
    print(f"\n [Libpcap] Starting time-window capture on {interface}...")
    
    # Open interface: max bytes, promiscuous, timeout
    cap = pcapy.open_live(interface, 65536, 1, 100)
    cap.setfilter("ip") # BPF filter at kernel level

    start_time = time.time()

    # dictionary to hold metrics per source ip
    ip_metrics = {}

    while (time.time() - start_time) < capture_duration:
        try:
            # Capture packet
            header, data = cap.next()
            if not data:
                continue
            
            # decode for stealth attack
            eth = dpkt.ethernet.Ethernet(data)

            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data
                src_ip = socket.inet_ntoa(ip.src)

                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data

                    # initialize ip metrics if not present
                    if src_ip not in ip_metrics:
                        ip_metrics[src_ip] = {
                            'total_packets': 0,
                            'total_bytes': 0,
                            'unique_dst_port': set(),
                            'syn_count': 0,
                            'ack_count': 0,
                            'curr_window_iats': []
                        }

                    # volumetric variables per ip
                    ip_metrics[src_ip]['total_packets'] += 1
                    ip_metrics[src_ip]['total_bytes'] += len(data)

                    # stealth variables per ip
                    ip_metrics[src_ip]['unique_dst_port'].add(tcp.dport)

                    if tcp.flags & dpkt.tcp.TH_SYN:
                        ip_metrics[src_ip]['syn_count'] += 1
                    if tcp.flags & dpkt.tcp.TH_ACK:
                        ip_metrics[src_ip]['ack_count'] += 1

                    # create a key for a set
                    flow_key = (ip.src, tcp.sport, ip.dst, tcp.dport)
                    packet_time = time.time()

                    if flow_key in flow_states:
                        iat = packet_time - flow_states[flow_key]
                        ip_metrics[src_ip]['curr_window_iats'].append(iat)

                    flow_states[flow_key] = packet_time


        except Exception:
            continue

# ----- compute metrics -------
    
    flow_data = []

    # compute metrics for each active ip
    for ip_address, metrics in ip_metrics.items():
        tot_pkts = metrics['total_packets']
        tot_bytes = metrics['total_bytes']

        # volumetric metrics
        flow_byts_s = tot_bytes / capture_duration
        flow_pkts_s = tot_pkts / capture_duration
        pkt_len_mean = (tot_bytes / tot_pkts) if tot_pkts > 0 else 0

        # stealth metrics
        port_scan_intensity = len(metrics['unique_dst_port'])
        syn = metrics['syn_count']
        ack = metrics['ack_count']
        syn_ack_ratio = syn / ack if ack > 0 else float(syn)

        # iat mean and iat std
        iats = metrics['curr_window_iats']
        iat_mean = float(np.mean(iats)) if iats else 0.0
        iat_std = float(np.std(iats)) if len(iats) > 1 else 0.0

        # add to flow data list
        flow_data.append({
            'Source_IP': ip_address,
            'flow_byts_s': flow_byts_s,
            'flow_pkts_s': flow_pkts_s,
            'pkt_len_mean': pkt_len_mean,
            'unique_ports': port_scan_intensity,
            'syn_ack_ratio': float(syn_ack_ratio),
            'iat_mean': iat_mean,
            'iat_std': iat_std
        })

    # return a dataframe
    return pd.DataFrame(flow_data)