from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import time 
import non_linear as nl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------------
# actual vs predicted plot for linear regression model
def actual_vs_predicted(ytest, predictions):
    y_true = np.array(ytest).flatten()
    y_pred = np.array(predictions).flatten()
    df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})

    sns.set_theme(style="whitegrid")
    
    g = sns.jointplot(
        data=df, x="Actual", y="Predicted",
        kind="reg", ci=None, height=7, ratio=4,
        scatter_kws={'alpha': 0.6, 's': 100, 'color': '#3498DB', 'edgecolor': 'w'},
        line_kws={'color': '#E74C3C', 'lw': 3},
        marginal_kws={'fill': True, 'color': '#3498DB', 'bins': 10}
    )

    ax = g.ax_joint
    
    all_data = np.concatenate([y_true, y_pred])
    margin = (all_data.max() - all_data.min()) * 0.1
    limit_min = all_data.min() - margin
    limit_max = all_data.max() + margin
    
    ax.set_xlim(limit_min, limit_max)
    ax.set_ylim(limit_min, limit_max)

    ax.plot([limit_min, limit_max], [limit_min, limit_max], 
            color='#2C3E50', linestyle='--', lw=2, alpha=0.5, label='Ideal Fit')

    g.fig.suptitle('Linear Regression Analysis', fontsize=14, fontweight='bold', y=1.02)
    ax.set_xlabel('Actual Entropy')
    ax.set_ylabel('Predicted Entropy')
    ax.legend()
    plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    
    plt.show()

 
# ---------------------------------------------------------------------------------

# residuals plot for linear regression model
def plot_residuals(ytest, predictions):
    y_true = np.array(ytest).flatten()
    y_pred = np.array(predictions).flatten()
    residuals = y_true - y_pred
    df_res = pd.DataFrame({'Predicted': y_pred, 'Residuals': residuals})

    sns.set_theme(style="whitegrid")
    
    g = sns.jointplot(
        data=df_res, x="Predicted", y="Residuals",
        kind="scatter", height=7, ratio=4,
        alpha=0.6, s=100, color='#2ECC71', edgecolor='w',
        marginal_kws={'fill': True, 'color': '#2ECC71'}
    )

    ax = g.ax_joint
    ax.axhline(y=0, color='#E74C3C', linestyle='--', lw=3, label='Zero Error')

    g.fig.suptitle('Residual Analysis: Checking for Patterns', fontsize=14, fontweight='bold', y=1.02)
    ax.set_xlabel('Predicted Entropy')
    ax.set_ylabel('Residuals (Actual - Predicted)')
    ax.legend()
    plt.savefig('residuals_plot.png', dpi=300, bbox_inches='tight')
    
    plt.show()

# ---------------------------------------------------------------------------------

# residuals distribution plot for linear regression model
def plot_residual_distribution(ytest, predictions):
    y_true = np.array(ytest).flatten()
    y_pred = np.array(predictions).flatten()
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    sns.histplot(residuals, kde=True, color='#9B59B6', bins=15, edgecolor='w')

    plt.axvline(x=0, color='#E74C3C', linestyle='--', lw=2, label='Mean Error = 0')
    
    plt.title('Distribution of Residuals (Error Normality)', fontsize=14, fontweight='bold')
    plt.xlabel('Residual Value (Error)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.savefig('residual_distribution.png', dpi=300, bbox_inches='tight')
    
    plt.show()


# ---------------------------------------------------------------------------------

def linear_regression():
    window_size = 20

    try:
        benign_dataset = pd.read_csv("capture.csv",sep=',')
        print("Dataset has been loaded... \n")
    except Exception as e:
        print("Error while loaded dataset!")
        return
    
    #get features from dataframe
    T_volume = benign_dataset['flow_byts_s']
    N_requests = benign_dataset['flow_pkts_s']
    S_len = benign_dataset['pkt_len_mean']

    #clean dataSeries
    T_volume[np.isinf(T_volume)] = np.mean(T_volume)
    N_requests[np.isinf(N_requests)] = np.mean(N_requests)
    S_len[np.isinf(S_len)] = np.mean(S_len)

    try:
        T_volume = T_volume.to_numpy()
        N_requests = N_requests.to_numpy()
        S_len = S_len.to_numpy()
    except Exception as e:
        print(e)
        return 
    
    #renyi entropy for each feature
    T_volume = nl.renyi_entropy(T_volume,window_size,10)
    N_requests = nl.renyi_entropy(N_requests,window_size,10)
    S_len = nl.renyi_entropy(S_len,window_size,10)

    #utilize scaler for data's scaling
    scaler = RobustScaler()

    T_volume = T_volume.reshape(-1,1)
    N_requests = N_requests.reshape(-1,1)
    S_len = S_len.reshape(-1,1)

    #datas for training
    X_train = np.column_stack([N_requests,S_len])
    y_train = np.column_stack([T_volume])

    X_train = scaler.fit_transform(X_train)
   
    #utilize model
    model = LinearRegression()

    #split dataset for training and testing
    xtrain, xtest, ytrain, ytest = train_test_split(X_train,y_train,test_size=0.20,random_state=42)

    print("Starting model's training ... \n")
    #block for training linear regression model
    try:
        start_time = time.time()
        model.fit(xtrain,ytrain)
        train_time = time.time() - start_time
        print(f"Model trained in {train_time:.5f} seconds")
    except Exception as e:
        print(f"Error while training model: {e}")
        return 

    score = model.score(xtest,ytest)
    print(f"\nModel's R-score is: {model.score(xtest,ytest):.3f} \n")


    #compute MSE (mean squared error) for evaluating model's performance
    predictions = model.predict(xtrain)
    for pred in range(len(predictions)):
        print(f"Predicted: {predictions[pred][0]:.3f}, Actual: {ytrain[pred][0]:.3f}")
        
    
    mae = np.mean(np.abs(predictions - ytrain))
    mse = np.mean((predictions - ytrain) ** 2)
    print(f"\nMean Absolute Error: {mae:.5f} \n")
    print("\n ----------------------------- \n")
    print(f"\nMean Squared Error: {mse:.5f} \n")


#compute μ_train and σ_train for calculating threshold for second stability check
    
    delta_train = np.abs(ytrain - predictions)
    mean_train = np.mean(delta_train)
    sigma_train = np.std(delta_train)
    print(f" \nMean of training residuals: {mean_train:.5f} \n")
    print(f" \nStandard Deviation of training residuals: {sigma_train:.5f} \n")



# ---------------------------------------------------------------------------------------------------------


    predictions_test = model.predict(xtest)
    actual_vs_predicted(ytest, predictions_test)
    plot_residuals(ytest, predictions_test)
    plot_residual_distribution(ytest, predictions_test)

    #block for saving model and scaler
    # try:
    #     print("\nSaving model ... \n")
    #     joblib.dump(model,'linear_regression.pkl')
    #     print("\nSaving scaler ...  \n")
    #     joblib.dump(scaler,'standardScaler.pkl')
    #     np.savez('train_metrics.npz', mean_train=mean_train, sigma_train=sigma_train)
    # except Exception as e:
    #     print(e)
    #     return 




   

if __name__ == "__main__":
    linear_regression()