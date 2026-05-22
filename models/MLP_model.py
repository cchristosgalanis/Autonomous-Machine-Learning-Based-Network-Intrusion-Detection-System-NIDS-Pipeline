import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
import os

def mlp_model():
    #load dataset
    data = np.load('features/final_features.npz')
    X, y = data['X'], data['y']

    #split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(
        hidden_layer_sizes=(128,64),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        verbose=True,
    )

    print("\n --- Training MLP Classifier --- \n")
    clf.fit(X_train, y_train)

    #evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))

    # save
    # os.makedirs('models', exist_ok=True)
    # joblib.dump(clf, 'models/nids_mlp_model.joblib')
    # joblib.dump(scaler, 'models/nids_scaler.joblib')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.show()

if __name__ == "__main__":
    mlp_model()