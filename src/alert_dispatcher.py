import json
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Any
from plyer import notification

# configure the logger for clean, professional output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_alert_config(config_path: str = "config/alerts_config.json") -> Optional[Dict[str, Any]]:
    """
    Loads the alert configuration from a JSON file.
    Returns a dictionary with the settings or None if an error occurs.
    """
    try:
        with open(config_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return None
    except json.JSONDecodeError:    
        logging.error(f"Failed to decode JSON from: {config_path}")
        return None

# ------------------------------------------------------------------------------------------------------------------

# function to create payload
def build_alert_payload(confidence: float, source_ip: str, action: str) -> Dict[str, Any]:
    """
    Creates the standardized JSON payload to be sent to external APIs.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
        "event_type": "INTRUSION_DETECTED",
        "confidence": round(confidence, 4),
        "action_taken": action.upper(),
        "target_ip": source_ip,
        "reason": f"NIDS model confidence hit {confidence * 100:.2f}% threshold."
    }

# ------------------------------------------------------------------------------------------------------------------

#function to send notification
def desktop_notification(title:str, message:str) -> None:
    """
    Sends a desktop notification using plyer.
    """
    try:
        notification.notify(
            title=title,
            message = message,
            app_name="NIDS Defense System",
            timeout=10  # seconds
        )
    except Exception as e:
        logging.error(f"Failed to send desktop notification: {e}")

# ------------------------------------------------------------------------------------------------------------------

def dispatch_alert(confidence: float, source_ip: str, config: Dict[str, Any]) -> None:
    """
    Evaluates the confidence score and routes the appropriate actions 
    (Block, Notify, or Ignore) based on the configured thresholds.
    """
    thresholds = config.get('thresholds', {})
    block_threshold = thresholds.get('critical_block', 0.97)
    notify_threshold = thresholds.get('critical_notify', 0.95)

    #  condiiton A 
    if confidence >= block_threshold:
        logging.warning(f"[BLOCK] Triggering defense for IP: {source_ip} (Confidence: {confidence*100:.1f}%)")
        payload = build_alert_payload(confidence, source_ip, "BLOCK")

        # post_to_firewall_api(payload, config)
        # send_notification(payload, config)

        desktop_notification(
            title="[CRITICAL] NIDS Auto-Block",
            message=f"Malicious payload blocked!\nSource: {source_ip}\nConfidence: {confidence*100:.1f}%"
        )

    # condition B
    elif confidence >= notify_threshold:
        logging.info(f"[NOTIFY] Suspicious activity from IP: {source_ip} (Confidence: {confidence*100:.1f}%)")
        payload = build_alert_payload(confidence, source_ip, "NOTIFY")

        # send_notification(payload, config)
        desktop_notification(
            title="[INFO] NIDS Alert",
            message=f"Suspicious activity detected!\nSource: {source_ip}\nConfidence: {confidence*100:.1f}%"
        )
    # CONDITION C -> normal traffic
    else:
        logging.debug(f"[NORMAL] Safe traffic from IP: {source_ip} (Confidence: {confidence*100:.1f}%)")