import pcapy
import pandas as pd
import time 
import dpkt

def libpcap_capture(capture_duration, interface):
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

    # return a dataframe
    flow_data = [{
        'flow_byts_s': flow_byts_s,
        'flow_pkts_s': flow_pkts_s,
        'pkt_len_mean': pkt_len_mean,
        'unique_ports': port_scan_intensity,
        'syn_ack_ratio': float(syn_ack_ratio)
    }]

    return pd.DataFrame(flow_data)