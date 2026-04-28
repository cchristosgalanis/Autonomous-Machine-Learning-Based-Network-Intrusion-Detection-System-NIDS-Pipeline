import numpy as np
import pandas as pd 
import joblib

#-------------------------------------------------------------------------------------------------


#Shannon entropy approach
def shannon_entropy(array,window_size):
    """
        argumentets: array -> array of CICFlowmeter output
                    window_size -> windowing (int number)
    
    """
    #checking for window size
    if window_size < 2:
        window_size = 1000;

    number_samples = int(np.floor(len(array)/window_size))
    entropy_shannon = np.zeros(number_samples)

    #because python enumarate arrays from 0
    for i in range(number_samples):
        #definition of the start and end boundaries
        start_idx = i * window_size;
        end_idx = start_idx + window_size;

        #calculating entropies
        temp_idx = array[start_idx : end_idx]

        temp = np.histogram(temp_idx,density=False,bins = 2 * int(np.ceil(window_size**(2/3)))) #using type for bins: k=2(window_size)^(3/2)
        count = temp[0]
        temp = count / np.sum(count)

        temp = temp[temp > 0] #to keep values that are not 0, to avoid inf values

        #shannont entropy approach
        entropy_shannon[i] = -np.sum(temp * np.log2(temp))

    #return array
    return entropy_shannon    

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
            #renyi entropy approach
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
    number_samples = int(np.floor(len(array)/window_size))
    mean_val = np.zeros(number_samples)
    std_val = np.zeros(number_samples)

    for i in range(number_samples):
        start_idx = i * window_size
        end_idx = start_idx + window_size

        temp_idx = array[start_idx : end_idx]
        temp = np.histogram(temp_idx,density=False,bins= 2 * int(np.ceil(window_size**(2/3))))
        count = temp[0]
        temp = count / np.sum(count)

        low = np.min(temp)
        high = np.max(temp)

        if low == high:
            print("Function returned an array with zeros")
            return array * 0
        else:
            norm_array = (temp - low) / (high - low)
            mean_val[i] = np.mean(norm_array)
            std_val[i] = np.std(norm_array)

    return mean_val, std_val


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
def first_stab_check(norm_array,live_traffic_array,window_size):
    """
        Vectorized stabillity check
    
        stabibility check for checking if live traffic diverge from mean value
        we compute: |T(t) - μ_t| > 0.1 * σ

        arguments: live_traffic_array -> numpy array (from CICFlowmeter)
                    norm_array -> normalized array 

        |T(t) - μ_t| = stabilitty
    """

    residual = np.zeros_like(live_traffic_array)

    for k in range(live_traffic_array.shape[1]):
        current_slice = live_traffic_array[:,k,:]

        residual[:,k,:] = np.abs(current_slice - norm_array)

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
def Z_score(residual):
    try:
        print("\n Loading training metrics ... \n")
        parameters = np.load('train_metrics_feature_expansion.npz')
        mean_train = parameters['mean_train']
        sigma_train = parameters['sigma_train']
    except Exception as e:
        print(f"Error loading training metrics: {e}")
        return None
    
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
def signature_analysis(analysis_batch,z_score):
    """
        analyzes the signature of the detected anomaly to distinguish
        between legitimate bursts and potential attacks
    """

    #calculate mean value of current anomalous window
    avg_bytes_s = np.mean(analysis_batch[:.0]) #flow_byts_s
    avg_pkts_s = np.mean(analysis_batch[:,1]) #flow_pkts_s
    avg_pkt_len = np.mean(analysis_batch[:,2]) #flow_len_mean

    #logic for signature identification
    # legitimate traffic
    if avg_bytes_s > (10*(10**5)) and avg_pkt_len > 1000:
        return "\n Legitimate Burst (Speedtest\Large Download) \n"
    
    # DDoS
    if avg_bytes_s > 1000 and avg_pkt_len < 150:
        return "\n Malicious Attack (Potential uDP/ICMP Flood) \n"
    
    #Port Scan
    if avg_pkts_s > 500 and avg_bytes_s < (50*(10**3)):
        return "\n Malicious Attack (Port Scanning) \n"
    
    return "\n Unknown Anomaly (Further check...) \n"










    