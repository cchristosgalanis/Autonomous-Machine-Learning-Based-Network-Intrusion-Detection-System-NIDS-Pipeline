import numpy as np
import pandas as pd 
import joblib
from collections import Counter

#-------------------------------------------------------------------------------------------------

#Shannon entropy approach
def shannon_entropy(domain_name):
    """"
        compute Shannon Entropy of a string (domain name)
        arguments: domain_name (string)
        returns: shannon entropy 
    """  

    if not domain_name:
        return 0.0
    
    # remove .com and generally everything after the first dot to focus on the main domain
    core_domain = domain_name.split('.')[0] if '.' in domain_name else domain_name

    length = len(core_domain)
    if length == 0:
        return 0.0
    
    #counter to compute instantaneous probability of each character in the domain name
    character_freq = Counter(core_domain)

    entropy = 0.0
    
    for count in character_freq.values():
        p_i = count / length
        entropy -= p_i * np.log2(p_i)

    return entropy

#-------------------------------------------------------------------------------------------------

#function to compute shannon entropy and get a result for domain names
def dns_entropy_check(domain_name, entropy_thresh=2.5):
    """"
        control function -> check entropies for each domain in order to be able to detect
        DGA domains or potential C2 communication attempts based on high entropy in domain names.
    """

    entropy = shannon_entropy(domain_name)
    malicious = entropy > entropy_thresh

    result = {
        "analyzed_domain": domain_name,
        "entropy": entropy,
        "is_DGA": malicious,
        "threshold": entropy_thresh
    }

    return result

#-------------------------------------------------------------------------------------------------

#Renyi entropy approach
def renyi_entropy(array,window_size,alpha):
    """
        arguments: array -> array of CICFlowmeter output
                    window_size -> windowing(int number)
                    alpha -> parameter for calculating Renyi entropy
    
    """
    if window_size < 2:
        window_size = 1000;

    number_samples = int(np.floor(len(array)/window_size))
    entropy_renyi = np.zeros(number_samples)

    for i in range(number_samples):
        start_idx = i * window_size
        end_idx = start_idx + window_size

        temp_idx = array[start_idx : end_idx]

        temp = np.histogram(temp_idx,density=False,bins= 2 * int(np.ceil(window_size**(2/3))))
        count = temp[0]
        temp = count / np.sum(count)

        temp = temp[temp > 0]

        if alpha == 1:
            #shannon entropy if alpha is 1
            entropy_renyi[i] = -np.sum(temp * np.log2(temp))
        else:
            #renyi entropy approachh
            entropy_renyi[i] = (1/(1-alpha)) * np.log2(np.sum(temp**alpha))

    #return array
    return entropy_renyi

#-------------------------------------------------------------------------------------------------

#statistical features beased on entropy of network
def mean_val(array):
    """
        arguments: array -> numpy array with entropy value
    """
    entropy_mean_val = np.mean(array,dtype=np.float64)
    return entropy_mean_val

def std_val(array):
    """
        arguments: array -> numpy array with entropy value
    """
    entropy_std_val = np.std(array,dtype=np.float64)
    return entropy_std_val


#-------------------------------------------------------------------------------------------------


#data normalization
def norm_datas(array,window_size):
    """
        arguments: array -> numpy array for each feature
                    window_size -> windowing in same size
        it is: x_norm = (x - x_min) / (x_max - x_min)
    """
    return np.mean(array),np.std(array)


#-------------------------------------------------------------------------------------------------


#function for windowing live traffic 
def windowing_live_function(live_traffic,window_size):
    number_samples = int(np.floor(len(live_traffic) / window_size))
    array = np.zeros((number_samples,window_size,live_traffic.shape[1]))

    for i in range(number_samples):
        start_idx = i * window_size
        end_idx = start_idx + window_size

        array[i] = live_traffic[start_idx : end_idx]

    return array

#-------------------------------------------------------------------------------------------------



#first residual check 
def first_stab_check(means,stds,live_traffic_array):
    """
        Vectorized stabillity check
    
        stabibility check for checking if live traffic diverge from mean value
        we compute: |T(t) - μ_t| > 0.1 * σ

        arguments: live_traffic_array -> numpy array (from CICFlowmeter)
                    norm_array -> normalized array 

        |T(t) - μ_t| = stabilitty
    """

    residual = np.abs(live_traffic_array - means)
    return residual

#-------------------------------------------------------------------------------------------------


#function to load linear regression model and scaler
def load_model_and_scaler():
    try:
        model = joblib.load('linear_regression_feat_exp.pkl')
        scaler = joblib.load('RobustScaler_feat_exp.pkl')
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None
    
#-------------------------------------------------------------------------------------------------

#function to load neural network model and scaler
def load_nn_model_and_scaler():
    import os
    
    active_model = 'models/mlp_model.joblib'
    active_scaler = 'models/scaler.joblib'
    base_model = 'models/nids_mlp_model2.joblib'
    base_scaler = 'models/nids_robust_scaler2.joblib'
    
    try:
        if os.path.exists(active_model) and os.path.exists(active_scaler):
            model_path = active_model
            scaler_path = active_scaler
        else:
            model_path = base_model
            scaler_path = base_scaler
            
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        print(f"Error loading neural network model or scaler: {e}")
        return None, None

#-------------------------------------------------------------------------------------------------


#second residual check | shannon entropy based with linear regression model
def entropy_based_stab_check(T_volume,N_requests,S_len,window_size,model,scaler):
    """"
        this function is for checking the residual of live traffic based on shannon entropy and linear regression model.
        Arguments: T_volume -> array of total volume of bytes per flow
                    N_requests -> array of total number of packets per flow
                    S_len -> array of total duration of flow
                    window_size -> windowing for calculating shannon entropy and linear regression model
"""

    #computing renyi entropy for each feature
    T_volume = renyi_entropy(T_volume,window_size,10)
    N_requests = renyi_entropy(N_requests,window_size,10) 
    S_len = renyi_entropy(S_len,window_size,10) 

    #reshaping arrays for linear regression model to (-1,1)
    T_volume = T_volume.reshape(-1,1)
    N_requests = N_requests.reshape(-1,1)
    S_len = S_len.reshape(-1,1)

    # --- Polnomial Expansion on Features ---
    N_requests_quad = np.square(N_requests)
    S_len_quad = np.square(S_len)

    Nreq_Slen = N_requests * S_len

    ones = np.ones_like(N_requests)

    #creating 1D vector with features for linear regression model
    X = np.column_stack([ones,N_requests,S_len,N_requests_quad,Nreq_Slen,S_len_quad])
    #using scaler for data's scaling
    X = scaler.transform(X)

    #predicting with linear regression model   
    T_volume_pred = model.predict(X)

    #calculating residual check
    entropy_residual = np.abs(T_volume - T_volume_pred)

    return entropy_residual


#-------------------------------------------------------------------------------------------------


#function for calculating Z-score for compute threshold for second stability check
def Z_score(residual,mean_train,sigma_train):
    z_score = (residual - mean_train) / sigma_train
    return z_score


#-------------------------------------------------------------------------------------------------


#function to calculate θ_cheb with Chebyshev's inequality for calculating threshold for second stability check
def theta_cheb(f_pos_rate):
    """
        arguments: f_pos_rate -> false positive rate for calculating threshold with Chebyshev's inequality

        1 / k^2 => k = sqrt(1 / f_pos_rate)
    """

    theta_chebysev = np.sqrt(1 / f_pos_rate)

    return theta_chebysev


#-------------------------------------------------------------------------------------------------

#function to compute dynamic windowing for live traffic
def volatility__dynamic_windowing(window_train,sigma_train,sigma_live):
    """
        this function is for calculating dynamic windowing size for second stability check based on volatility of live traffic and training data.
        Arguments: window_train -> windowing size for training data
                    sigma_train -> standard deviation of training data
                    sigma_live -> standard deviation of live traffic

        volatility = sigma_live / sigma_train
        dynamic_window_size = window_train * volatility
    """

    volatility = sigma_live / sigma_train
    dynamic_window_size = int(window_train * volatility)

    return dynamic_window_size


#-------------------------------------------------------------------------------------------------

#function for signature analysis to seperate Speedtest/LargeFile downlod (ligitimate traffic)
#and DDoS attacks / Port Scan
def signature_analysis(analysis_batch,raw_mean,raw_std):
    """
        analyzes the signature of the detected anomaly to distinguish
        between legitimate bursts and potential attacks
    """

    avg_vals = np.mean(analysis_batch,axis=0)
    k = 5

    """
        0 -> bytes_s
        1 -> pkts_s
        2 -> pkt_len_mean
    """
    byte_threshold = raw_mean[0] + (k * raw_std[0])
    pkt_len_threshold = raw_mean[2] + (k * raw_std[2])

    if avg_vals[0] > byte_threshold and avg_vals[2] > pkt_len_threshold:
        return "Legitimate Burst (Speedtest/Large Download)"

    if avg_vals[1] > (raw_mean[1] + 5 * raw_std[1]) and avg_vals[2] < raw_mean[2]:
        return "Malicious Attack (Potential Flood)"
    
    return "Unknown Anomaly (Further check...)"


#-------------------------------------------------------------------------------------------------


#function for kinematic analysis
def compute_kinematic(residual_memory):
    """
        compute 1st (velocity) and 2nd (acceleration) derivatives on residual
        arguments: list with 2 or 3 last residuals
        returns: velocity, acceleration
    """

    # less than 2 samples, cannot compute velocity
    if len(residual_memory) < 2:
        return 0.0, 0.0
    
    # exactly 2 samples, compute only velocity (acceleration is 0)
    if len(residual_memory) == 2:
        R_t1 = residual_memory[0] #(t-1)
        R_t = residual_memory[1]  #(t)
        velocity = R_t - R_t1
        return velocity, 0.0

    # exactly 3 samples, compute both
    R_t2 = residual_memory[0] #(t-2)
    R_t1 = residual_memory[1] #(t-1)
    R_t = residual_memory[2]  #(t)

    V_old = R_t1 - R_t2
    V_new = R_t - R_t1

    # 2nd derivative -> R(t) - 2R(t-1) + R(t-2)
    acceleration = V_new - V_old

    return V_new, acceleration


#-------------------------------------------------------------------------------------------------

#function for spatial correlation analysis using MinHash signatures
def spatial_jaccard(curr_sig, prev_sig,k,threshold=0.70):
    """"
        compute Jaccard similarity between current and previous MinHash signatures
        arguments: curr_sig -> current MinHash signature
                    prev_sig -> previous MinHash signature
                    k -> size of MinHash signature
        returns: Jaccard similarity score and boolean indicating if similar (above threshold)
    """

    if np.all(curr_sig == -1) or np.all(prev_sig == -1):
        return {
            "is_botnet": False,
            "jaccard_score": 0.0,
            "matches": 0,
            "status": "idle"
        }
    
    # vectorized Jaccard similarity calculation
    matches = np.sum(curr_sig == prev_sig)
    jaccard_score = matches / k

    # decision based on threshold
    is_botnet = jaccard_score >= threshold

    return {
        "is_botnet": bool(is_botnet),
        "jaccard_score": float(jaccard_score),
        "matches": int(matches),
        "status": "potentially malicious" if is_botnet else "benign"
    }








    