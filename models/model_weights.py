import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load model
model_path = 'nids_mlp_model3.joblib'
mlp_model = joblib.load(model_path)

#get model weigths
weights_layer1 = mlp_model.coefs_[0]

feature_names = ['Velocity (v)', 'Acceleration (a)', 'IAT Mean', 'SA Ratio', 'Jaccard Score', 'Shannon Entropy (DNS)']

plt.figure(figsize=(12,8))
sns.set_theme(style='white')

ax = sns.heatmap(
    weights_layer1, 
    annot=False,        
    cmap="RdBu_r",     
    center=0,             
    yticklabels=feature_names,
    xticklabels=False,
    cbar_kws={'label': 'Weight Magnitude'}
)

plt.title("MLP Neural Network: First Hidden Layer Weights",fontsize=16,fontweight='bold',pad=20)
plt.ylabel("Input Features", fontsize=14,fontweight='bold')
plt.xlabel('Hidden Layer 1 Neurons',fontsize=14,fontweight='bold')

plt.xticks(rotation=45,ha='right')
plt.yticks(rotation=0,fontsize=12)

plt.tight_layout()

plt.savefig('mlp_weights_heatmap.png',dpi=300,bbox_inches='tight')
plt.show()