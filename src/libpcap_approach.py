import pcapy
import pandas as pd
import time 
import dpkt
import numpy as np
import socket
import struct

def libpcap_capture(capture_duration, interface, flow_states,target_port=80):
    print(f"\n [Libpcap] Starting time-window capture on {interface}...")
    
    # Open interface: max bytes, promiscuous, timeout
    cap = pcapy.open_live(interface, 65536, 1, 100)
    cap.setfilter("ip") # BPF filter at kernel level

    start_time = time.time()

    # dictionary to hold stealth metrics per source ip
    ip_metrics = {}

    # initialize global parameters for minHash features
    # initialize k for minHash (number of hash functions)
    k = 128
    # biggest prime number for hashing (for minHash)
    prime_number = 4294967311 # because biggest number is for IPv4 is 2^32-1, we take the biggest prime number below that for better hash distribution
    #initialize random hash functions for minHash
    np.random.seed(42) # for reproducibility
    a_coeffs = np.random.randint(1, prime_number, size=k,dtype=np.int64) # k-random coefficients for hash functions
    b_coeffs = np.random.randint(0, prime_number, size=k,dtype=np.int64) # k-random coefficients for hash functions
    # initialize signature matrix for minHash (k rows for k hash functions, 1 column for each unique destination port)
    signature_matrix = np.full(k,np.inf) # initialize with infinity for minHash
    
    # global counters for volumetric features
    global_total_packets = 0
    global_total_bytes = 0

    # list for raw DNS queries in this window (for potential future use in spatial correlation)
    dns_queries = []

    while (time.time() - start_time) < capture_duration:
        try:
            # Capture packet
            header, data = cap.next()
            if not data:
                continue
            
            # decode frame
            eth = dpkt.ethernet.Ethernet(data)

            if isinstance(eth.data, dpkt.ip.IP):
                ip = eth.data
                
                # update global volumetric counters
                global_total_packets += 1
                global_total_bytes += len(data)
                
                src_ip = socket.inet_ntoa(ip.src)

                # analyze TCP packets for stealth metrics
                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data

                    # minHash signature update for this source IP based on destination port (for port scan detection)
                    if tcp.dport == target_port:
                        ip_int = struct.unpack("!I", ip.src)[0] # convert source IP to integer for hashing

                        current_hashes = (a_coeffs * ip_int + b_coeffs) % prime_number # compute k hash values for this IP using the random coefficients
                        signature_matrix = np.minimum(signature_matrix, current_hashes) # update the minHash signature

                    # initialize ip metrics if not present
                    if src_ip not in ip_metrics:
                        ip_metrics[src_ip] = {
                            'unique_dst_port': set(),
                            'syn_count': 0,
                            'ack_count': 0,
                            'curr_window_iats': []
                        }

                    # stealth variables per ip
                    ip_metrics[src_ip]['unique_dst_port'].add(tcp.dport)

                    if tcp.flags & dpkt.tcp.TH_SYN:
                        ip_metrics[src_ip]['syn_count'] += 1
                    if tcp.flags & dpkt.tcp.TH_ACK:
                        ip_metrics[src_ip]['ack_count'] += 1

                    # create a key for stateful tracking
                    flow_key = (ip.src, tcp.sport, ip.dst, tcp.dport)
                    packet_time = time.time()

                    if flow_key in flow_states:
                        iat = packet_time - flow_states[flow_key]
                        ip_metrics[src_ip]['curr_window_iats'].append(iat)

                    flow_states[flow_key] = packet_time
                
                # capture DNS queries for potential future use in spatial correlation
                elif isinstance(ip.data, dpkt.udp.UDP):
                    udp = ip.data
                    
                    #check if destination port is 53 (DNS)
                    if udp.dport == 53:
                        try:
                            #parse DNS payload
                            dns = dpkt.dns.DNS(udp.data)

                            # if query (qr==0) AND questions present
                            if dns.qr == 0 and len(dns.qd) > 0:
                                qname = dns.qd[0].name # raw domain name from DNS query

                                dns_queries.append({
                                    "timestamp": time.time(),
                                    "source_ip": src_ip,
                                    "domain": qname
                                })
                        except Exception:
                            pass

        except Exception:
            continue

    # avoid stale flow states (older than 60 seconds) to prevent memory bloat
    current_time = time.time()
    for key in list(flow_states.keys()):
        if current_time - flow_states[key] > 60.0:
            del flow_states[key]

    # ----- compute metrics -------
    
    flow_data = []

    # compute global volumetric features (same for all IPs in this window)
    global_flow_byts_s = global_total_bytes / capture_duration
    global_flow_pkts_s = global_total_packets / capture_duration
    global_pkt_len_mean = (global_total_bytes / global_total_packets) if global_total_packets > 0 else 0

    # compute metrics for each active ip
    for ip_address, metrics in ip_metrics.items():
        
        # stealth metrics
        port_scan_intensity = len(metrics['unique_dst_port'])
        syn = metrics['syn_count']
        ack = metrics['ack_count']
        
        # avoid division by zero and set a cap for pure SYN floods
        if ack > 0:
            syn_ack_ratio = syn / ack
        elif syn > 0:
            syn_ack_ratio = 999.0  # temporary cap for pure SYN floods | can be tuned or set to float('inf')
        else:
            syn_ack_ratio = 0.0

        # iat mean and iat std
        iats = metrics['curr_window_iats']
        iat_mean = float(np.mean(iats)) if iats else 0.0
        iat_std = float(np.std(iats)) if len(iats) > 1 else 0.0

        # add to flow data list
        flow_data.append({
            'Source_IP': ip_address,
            'flow_byts_s': global_flow_byts_s,   # getting the global bytes per second for this window  
            'flow_pkts_s': global_flow_pkts_s,   # getting the global packets per second for this window  
            'pkt_len_mean': global_pkt_len_mean, # getting the global mean packet length for this window  
            'unique_ports': port_scan_intensity,
            'syn_ack_ratio': float(syn_ack_ratio),
            'iat_mean': iat_mean,
            'iat_std': iat_std
        })

    signature_matrix[np.isinf(signature_matrix)] = -1

    spatial_payload = {
        "timestamp": time.time(),
        "target_port": target_port,
        "k_size": k,
        "signature": signature_matrix.tolist() # convert numpy array to list for JSON serialization
    }

    # return a dataframe
    return pd.DataFrame(flow_data), spatial_payload, dns_queries