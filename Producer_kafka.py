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

    try:
        while True:
            #call libpcap function from libpcap_approach file to sniff network
            #check for arguments in function libpcap_capture
            flow_data = libpcap_capture(capture_duration=1,interface='en0')

            json_string = json.dumps(flow_data)
            bytes = json_string.encode('utf-8')

            #send to Kafka
            producer.produce(topic=topic_name, value=bytes, callback=callback)

            producer.poll(0)
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n Stopping Sniffer ... \n")

    producer.flush()
    print("\n All messages have been sent successfully \n")

if __name__ == "__main__":
    Producer_func()


