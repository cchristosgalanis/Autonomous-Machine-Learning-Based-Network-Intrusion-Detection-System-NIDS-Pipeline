import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.neural_network import MLPClassifier

def plot_learning_curves(clf):
    # plot loss and validation score curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # loss curve
    axes[0].plot(clf.loss_curve_, color='blue', linewidth=2)
    axes[0].set_title('Training Loss Curve')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # validation score curve
    if hasattr(clf, 'validation_scores_'):
        axes[1].plot(clf.validation_scores_, color='green', linewidth=2)
        axes[1].set_title('Validation Accuracy Curve')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy Score')
        axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_conf_matrix(y_test, y_pred):
    # plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def plot_roc(y_test, y_prob):
    # plot roc curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_pr_curve(y_test, y_prob):
    # plot precision-recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {ap_score:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def mlp_model():
    # load dataset
    data = np.load('features/final_features.npz')
    X, y = data['X'], data['y']

    # split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # init model
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=10,
        verbose=True,
    )

    print("\n--- Training MLP Classifier ---\n")
    clf.fit(X_train, y_train)

    # evaluate
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---\n")
    print(classification_report(y_test, y_pred))

    # generate plots
    plot_learning_curves(clf)
    plot_conf_matrix(y_test, y_pred)
    plot_roc(y_test, y_prob)
    plot_pr_curve(y_test, y_prob)

    # save models
    # os.makedirs('models', exist_ok=True)
    # joblib.dump(clf, 'models/nids_mlp_model.joblib')
    # joblib.dump(scaler, 'models/nids_scaler.joblib')

if __name__ == "__main__":
    mlp_model()