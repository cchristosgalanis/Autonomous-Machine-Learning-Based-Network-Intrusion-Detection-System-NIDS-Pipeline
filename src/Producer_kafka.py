from confluent_kafka import Producer
import json
import time 
import argparse
import os
from libpcap_approach import libpcap_capture

def callback(err,msg):
    if err is not None:
        print(f"\n Message delivery failed ... \n")
    else:
        msg.partition()
        pass

def Producer_func():

    default_iface = os.getenv('DEFAULT_IFACE', 'en0')

    parser = argparse.ArgumentParser(description='Kafka Producer for Network Flow Data')
    parser.add_argument('-i','--iface',
                        type=str,
                        default='en0',
                        help='Network interface to capture traffic from (default: en0)'
                )

    args = parser.parse_args()
    interface = args.iface

    conf = {
        'bootstrap.servers':'localhost:9092'
    }

    producer = Producer(conf)

    # initialize 2-different topics
    topic_name = 'live-network-flows'
    spatial_topic = 'spatial-minhash-signatures'
    active_flow_states = {}

    try:
        while True:
            #call libpcap function from libpcap_approach file to sniff network
            #check for arguments in function libpcap_capture
            flow_data, spatial_payload = libpcap_capture(capture_duration=2, interface=interface, flow_states=active_flow_states)

            if not flow_data.empty:
                # iterate over all rows (all IPs)
                for index, row in flow_data.iterrows():
                    flow_dict = {
                        "Source_IP": str(row['Source_IP']),
                        "T_vol": float(row['flow_byts_s']),
                        "N_req": float(row['flow_pkts_s']),
                        "S_len": float(row['pkt_len_mean']),
                        'U_ports': int(row['unique_ports']),
                        'SA_ratio': float(row['syn_ack_ratio']),
                        'IAT_mean': float(row['iat_mean']),
                        'IAT_std': float(row['iat_std'])
                    }

                    json_string = json.dumps(flow_dict)
                    bytes = json_string.encode('utf-8')

                    #send to Kafka
                    producer.produce(topic=topic_name, value=bytes, callback=callback)
            
            if spatial_payload and len(spatial_payload.get('signature',[])) > 0:
                spatial_json = json.dumps(spatial_payload)
                spatial_bytes = spatial_json.encode('utf-8')
                producer.produce(topic=spatial_topic, value=spatial_bytes, callback=callback)

            producer.poll(0)
            
    except KeyboardInterrupt:
        print("\n Stopping Sniffer ... \n")

    producer.flush()
    print("\n All messages have been sent successfully \n")

if __name__ == "__main__":
    Producer_func()