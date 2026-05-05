from confluent_kafka import Producer
import json
import time 
from libpcap_approach import libpcap_capture

def callback(err,msg):
    if err is not None:
        print(f"\n Message delivery failed ... \n")
    else:
        msg.partition()
        pass

def Producer_func():
    conf = {
        'bootstrap.servers':'localhost:9092'
    }

    producer = Producer(conf)

    topic_name = 'live-network-flows'
    active_flow_states = {}

    try:
        while True:
            #call libpcap function from libpcap_approach file to sniff network
            #check for arguments in function libpcap_capture
            flow_data = libpcap_capture(capture_duration=1,interface='en0',flow_states=active_flow_states)

            if not flow_data.empty:
                row = flow_data.iloc[0]

                flow_dict = {
                    "T_vol": float(row['flow_byts_s']),
                    "N_req": float(row['flow_pkts_s']),
                    "S_len": float(row['pkt_len_mean']),
                    'U_ports': int(row['unique_ports']),
                    'SA_ratio': float(row['syn_ack_ratio'])
                }

                json_string = json.dumps(flow_dict)
                bytes = json_string.encode('utf-8')

                #send to Kafka
                producer.produce(topic=topic_name, value=bytes, callback=callback)

                producer.poll(0)
    except KeyboardInterrupt:
        print("\n Stopping Sniffer ... \n")

    producer.flush()
    print("\n All messages have been sent successfully \n")

if __name__ == "__main__":
    Producer_func()


