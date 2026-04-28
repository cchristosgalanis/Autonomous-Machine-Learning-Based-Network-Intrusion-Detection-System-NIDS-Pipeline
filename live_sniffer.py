from scapy.all import conf, sniff,IP,TCP,UDP
import pandas as pd
import time

#function to sniff live traffic and extract features
conf.use_pcap = True

def capture_live_traffic(capture_duration, interface):
    """"
        listens to the network of {capture_duration} seconds. 
        arguments: capture_duration -> duration of sniffing in seconds
                   interface -> network interface to sniff on (default is 'en0' for macOS)}
        
                   Groups packets into Flows and returns a pandas Dataframe with the features: flow_byts_s, flow_pkts_s, pkt_len_mean
    """

    print(f"\n Starting live traffic capture for {capture_duration} seconds on intercace {interface} ... \n")
    flow_hash_map = {}

    def process_packet(packet):
        #if there is IP layer in packet
        if IP in packet:
            src_ip = packet[IP].src
            dst_ip = packet[IP].dst
            protocol = packet[IP].proto
            length = len(packet)

            src_port = 0
            dst_port = 0
            if TCP in packet:
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
            elif UDP in packet:
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport

            #creating unique key (bidirectional flow classifier)
            ips = tuple(sorted([src_ip, dst_ip]))
            ports = tuple(sorted([src_port, dst_port]))
            flow_key = (ips[0],ips[1],ports[0],ports[1],protocol)

            #update hash map
            if flow_key not in flow_hash_map:
                flow_hash_map[flow_key] = {'packets':0, 'bytes':0}
            
            flow_hash_map[flow_key]['packets'] += 1
            flow_hash_map[flow_key]['bytes'] += length

    sniff(iface=interface, prn=process_packet, store=False,timeout=capture_duration,
          promisc=False,filter="ip")

    print("\n Live traffic capture completed. Processing flows ... \n")
    flow_data = []
    for key, metrics in flow_hash_map.items():
        pkts = metrics['packets']
        byts = metrics['bytes']

        #same as CICFlowmeter
        flow_pkts_s = pkts / capture_duration
        flow_byts_s = byts / capture_duration

        pkt_len_mean = byts / pkts

        flow_data.append({
            'flow_byts_s': flow_byts_s,
            'flow_pkts_s': flow_pkts_s,
            'pkt_len_mean': pkt_len_mean
        })

    df = pd.DataFrame(flow_data)

    return df
