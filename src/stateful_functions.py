import numpy as np

#----------------------------------------------------------------------------------------------------------------------------


#function to checkk stealth port scanning
def port_scan(u_ports, port_threshold=50):
    """
        stealth port scanning (e.g nmap -sS)
    """

    if u_ports > port_threshold:
        return True
    return False


#----------------------------------------------------------------------------------------------------------------------------


#function to check syn flood attack
def syn_flood(sa_ratio, ratio_threshold=10.0):
    """
        check half-open connections and stealth SYN Floods
        under benign traffic SA_ratio (SYN/ACK) lim to -> 0 or 1
        but under attack this limit goes to inf
    """

    if sa_ratio > ratio_threshold:
        return True
    return False 

#----------------------------------------------------------------------------------------------------------------------------


#function to check slowloris attacks
def slowloris(iat_mean,iat_std, mean_threshold = 3.0, std_threshold = 0.5):
    """
        check on application layer
    """
    #adding this check in order to avoid false positives when there is no traffic at all (iat_mean and iat_std are 0)
    if iat_std <= 0.0001:
        return False

    if iat_mean > mean_threshold and iat_std < std_threshold:
        return True
    return False


#----------------------------------------------------------------------------------------------------------------------------


def stealth_signature_analysis(u_ports,sa_ratio,iat_mean,iat_std):
    #port scan check
    if port_scan(u_ports):
        return "Stealth Port Scan (Reconnaissance Phase)"
    
    #asymmetric State Exhaustion
    elif syn_flood(sa_ratio):
        return "Stealth SYN Flood (State Exhaustion)"
    
    #slowloris
    elif slowloris(iat_mean,iat_std):
        return "Slowloris (Application-Layer DoS)"
    
    return "Normal"