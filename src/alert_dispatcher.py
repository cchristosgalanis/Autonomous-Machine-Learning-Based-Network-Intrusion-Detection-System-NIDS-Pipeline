import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#function to load alert configuration from a JSON file
def load_conf(conf_path="config/alerts_config.json"):
    try:
        with open(conf_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {conf_path}")
        return None
    except json.JSONDecodeError:    
        logging.error(f"Error decoding JSON from the configuration file at {conf_path}")
        return None
    
# ------------------------------------------------------------------------------------------------------------------

# function to create payloads for API calls
def create_payload(confidence, source_ip, action):
    return {
        "timestamp": datetime.utcnow().isoformat() + 'Z',
        "event_type": "Intrusion_Detected",
        "confidence": confidence,
        "action_taken" : action,
        "target_ip": source_ip,
        "reason": f"NIDS model confidence hit {confidence*100}% threshold."
    }
      

# ------------------------------------------------------------------------------------------------------------------


# fdefault function to dispatch alerts based on the configuration
def dispatch_function(confidence,source_ip,config):
    thresholds = config.get('thresholds', {})

    # critical threshold Block A 
    if confidence >= thresholds.get('critical_block', 0.97):
        logging.info(f" Block condition for IP: {source_ip} (Confidence: {confidence})")
        payload = create_payload(confidence, source_ip, "Block")

        # function for POST request to firewall API
        # post_to_firewall_api(payload,config)

        # function for POST request to Slack/Dashboard 
        # send_notific(payload,config)

    elif confidence >= thresholds.get('critical_notify', 0.95):
        logging.info(f" Notify condition for IP: {source_ip} (Confidence: {confidence})")
        payload = create_payload(confidence, source_ip, "Notify")

        # function for POST request to Slack/Dashboard 
        # send_notific(payload,config)
    
    else:
        logging.info(f" No action for IP: {source_ip} (Confidence: {confidence})")

    