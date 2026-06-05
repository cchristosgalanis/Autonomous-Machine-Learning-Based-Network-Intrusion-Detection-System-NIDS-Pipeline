import numpy as np
import psycopg2
import os

def sanitization_logic():
    """
    Sanitizes raw training data by querying the database directly.
    - Fetches only unprocessed records (is_processed=FALSE).
    - Uses SQL to filter only highly confident predictions or Host traffic.
    - Labels Host traffic as Benign (0).
    - Updates the database to mark fetched records as processed.
    """

    HOST_IP = os.getenv('HOST_IP', '127.0.0.1')
    
    # Database Credentials
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASS = os.getenv('DB_PASS', 'yourpassword')
    DB_NAME = os.getenv('DB_NAME', 'postgres')
    
    T_SAFE = 0.15
    CONFIDENCE_THRESHOLD = 0.85
    
    sanitized_X = []
    sanitized_y = []

    try:
        # connect to the database
        conn = psycopg2.connect(host=DB_HOST, user=DB_USER, password=DB_PASS, dbname=DB_NAME)
        cursor = conn.cursor()

        # get only unprocessed records with high confidence or Host traffic
        select_query = f"""
            SELECT source_ip, velocity, acceleration, iat_mean, sa_ratio, jaccard_score, entropy_shannon, prediction 
            FROM network_traffic_events 
            WHERE is_processed = FALSE 
              AND (probability <= {T_SAFE} OR probability >= {CONFIDENCE_THRESHOLD} OR source_ip = '{HOST_IP}');
        """
        cursor.execute(select_query)
        records = cursor.fetchall()

        if not records:
            cursor.close()
            conn.close()
            return None, None

        # feature extraction and labeling
        for row in records:
            ip = row[0]
            features = list(row[1:7])  # mapping velocity, acceleration, iat_mean, sa_ratio, jaccard_score, entropy_shannon
            
            # if host_ip -> benign = 0 | else use the original prediction 
            label = 0 if ip == HOST_IP else int(row[7])
            
            sanitized_X.append(features)
            sanitized_y.append(label)

        # update database for this batch to prevent reprocessing
        update_query = "UPDATE network_traffic_events SET is_processed = TRUE WHERE is_processed = FALSE;"
        cursor.execute(update_query)
        conn.commit()

        cursor.close()
        conn.close()

        return np.array(sanitized_X), np.array(sanitized_y)

    except Exception as e:
        print(f"\n Database Sanitization Error: {e}")
        return None, None
