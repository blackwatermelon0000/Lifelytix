import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, log_loss
import numpy as np

def run_neural_network(df):
    df = df.dropna(subset=['BMI', 'Age', 'Waist', 'Gender'])
    df['Gender_encoded'] = LabelEncoder().fit_transform(df['Gender'])
    df['BMICategory'] = pd.qcut(df['BMI'], q=4, labels=[0, 1, 2, 3])

    X = df[['Age', 'Waist', 'Gender_encoded']]
    y = df['BMICategory'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_train_pred = model.predict(X_train_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    report = classification_report(y_test, y_pred, output_dict=True)
    acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    logloss = log_loss(y_test, y_proba)
    auc_ovr = roc_auc_score(label_binarize(y_test, classes=[0, 1, 2, 3]), y_proba, multi_class='ovr')
    cross_val = cross_val_score(model, X, y, cv=5)

    # Confusion matrix plot
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    fig_cm.patch.set_facecolor('#1E1E1E')
    ax_cm.set_facecolor('#1E1E1E')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=True, square=True, ax=ax_cm,
                annot_kws={"size": 9, "color": "black"})
    ax_cm.set_xlabel("Predicted", fontsize=8, color='white')
    ax_cm.set_ylabel("Actual", fontsize=8, color='white')
    ax_cm.tick_params(colors='white', labelsize=8)
    for spine in ax_cm.spines.values():
        spine.set_edgecolor("#FFBF00")

    return model, {
        "accuracy_test": acc,
        "accuracy_train": train_acc,
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1-score": report['weighted avg']['f1-score'],
        "auc_roc": auc_ovr,
        "log_loss": logloss,
        "cross_val_mean": cross_val.mean(),
        "cross_val_std": cross_val.std(),
        "report": report,
        "X_test": X_test_scaled,
        "y_proba": y_proba
    }, fig_cm
