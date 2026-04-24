import pandas as pd
import matplotlib.pyplot as plt
import non_linear as nl
import numpy as np
import time 


def main():
    try:
        benign_dataset = pd.read_csv("benign.csv",sep=',')
        print("Dataset has been loaded... \n")
    except Exception as e:
        print("Error while loaded dataset!")
        return
    
    #get features from dataframe
    T_volume = benign_dataset['flow_byts_s']
    N_requests = benign_dataset['flow_pkts_s']
    S_len = benign_dataset['pkt_len_mean']

    #transforming dataframes to np.arrays
    try:
        T_volume = T_volume.to_numpy()
        N_requests = N_requests.to_numpy()
        S_len = S_len.to_numpy()
    except RuntimeError as e:
        print(e)
        return 

#-------------------------------------------------------------------------------------------------------

    start_time = time.time()

    #creating 1D vector with live traffic
    live_traffic = np.column_stack((T_volume,N_requests,S_len))

    #compute σ for live traffic
    sigma_live = np.std(live_traffic, axis=0)
    parameters = np.load('train_metrics.npz')
    sigma_train = parameters['sigma_train']
    window_size = nl.volatility__dynamic_windowing(window_train=20,sigma_train=sigma_train,sigma_live=sigma_live)

    print(window_size)

    # datas normalization to [0,1]
    """
        this block represents the μ and σ of paper. Show now with this, I have calculated these values and 
        we can continue with residual check
    """
    flow_byts_mean,flow_byts_std = nl.norm_datas(T_volume,window_size)
    flow_pkts_mean,flow_pkts_std = nl.norm_datas(N_requests,window_size)
    flow_dur_mean,flow_dur_std = nl.norm_datas(S_len,window_size)
   

    #create 1D vector with this normalized features
    norm_flows_mean = np.column_stack([flow_byts_mean,flow_pkts_mean,flow_dur_mean])
    norm_flows_std = np.column_stack([flow_byts_std,flow_pkts_std,flow_dur_std])

    #function for windowing live traffic
    live_traffic = nl.windowing_live_function(live_traffic,window_size)

    #creating threshold for each stabilities checks
    threshold1_f = 0.1 * norm_flows_std

    #get residual check
    print("\n --- Stability Check: --- \n")
    try:
        residual = nl.first_stab_check(norm_flows_mean,live_traffic,window_size=window_size)
        print("\n First Stabiblity check has been calculated ... \n")
    except RuntimeError as e:
        print(e)
        print("\n Something went wrong this calculation! \n")
        return 
    
    if np.any(residual > threshold1_f[:,np.newaxis,:]):
        print("\n There might be unstable traffic! \n")
        print("\n We have to check further more for anomalies and anomalies classification! \n")
        flag1 = True
    else:
        print("Everything is normal!")
        flag1 = False


#-------------------------------------------------------------------------------------------------------

    #checking further more with shannon entropy and linear regression model
    if flag1 == True:
        try:
            model, scaler = nl.load_model_and_scaler()

            if model is None or scaler is None:
                print("Model or scaler could not be loaded. Exiting.")
                return
            else:
                print("Model and Scaler loaded successfully.")

            entropy_residual = nl.entropy_based_stab_check(T_volume,N_requests,S_len,window_size,model,scaler)
            print("\n Entropy based residual check has been calculated ... \n")
        except RuntimeError as e:
            print(e)
            print("\n Something went wrong this calculation! \n")
            return 
        
        #calculate Z_score threshold and then using 3-sigma rule for checking if there is threshold violation
        z_score = nl.Z_score(entropy_residual)
        theta_cheb = nl.theta_cheb(0.02)

        if np.any(z_score > theta_cheb): 
            print("\n There might be anomalous traffic! \n")
            end_time = time.time() - start_time
            print(f"\n Time taken for the whole process: {end_time:.4f} seconds \n")
        else:
            print("\n Everything is normal! \n")
            end_time = time.time() - start_time
            print(f"\n Time taken for the whole process: {end_time:.4f} seconds \n")


#-------------------------------------------------------------------------------------------------------

    
if __name__ == "__main__":
    main()




