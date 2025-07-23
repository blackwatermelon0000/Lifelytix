import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

def train_logistic_model(df):
    df = df.dropna(subset=['BMI', 'Age', 'Waist', 'Gender'])
    df['Gender_encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})

    if df['BMI'].nunique() < 4:
        return None, None, {}, plt.figure(), plt.figure(), plt.figure()

    try:
        df['Category'] = pd.qcut(df['BMI'], q=4, labels=[0, 1, 2, 3])
    except ValueError:
        return None, None, {}, plt.figure(), plt.figure(), plt.figure()

    X = df[['Age', 'Waist', 'Gender_encoded']]
    y = df['Category'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, multi_class='ovr')
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_proba)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')

    metrics = {
        "accuracy": acc,
        "precision": report_dict['weighted avg']['precision'],
        "recall": report_dict['weighted avg']['recall'],
        "f1-score": report_dict['weighted avg']['f1-score'],
        "log_loss": logloss,
        "roc_auc": roc_auc
    }

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    fig_cm.patch.set_facecolor('#1E1E1E')  # match Streamlit dark background
    ax_cm.set_facecolor('#1E1E1E')

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='coolwarm',
        # linewidths=0.5,
        # linecolor='black',
        cbar=True,
        square=True,
        ax=ax_cm,
        annot_kws={"size": 12, "color": "black"}
    )
    ax_cm.set_xlabel("Predicted", fontsize=10, color='white')
    ax_cm.set_ylabel("Actual", fontsize=10, color='white')
    ax_cm.tick_params(axis='both', colors='white')

    for spine in ax_cm.spines.values():
        spine.set_edgecolor("#FFBF00")

    # --- Feature Importance ---
    coef_df = pd.DataFrame(model.coef_, columns=X.columns)
    coef_df = coef_df.T
    coef_df.columns = [f'Class {i}' for i in coef_df.columns]

    fig_imp, ax_imp = plt.subplots(figsize=(6, 4), dpi=120)
    fig_imp.patch.set_facecolor('#1E1E1E')
    ax_imp.set_facecolor('#1E1E1E')

    coef_df.plot(
        kind='bar',
        ax=ax_imp,
        color=sns.color_palette("colorblind", n_colors=len(coef_df.columns))
    )
    ax_imp.set_xticklabels(ax_imp.get_xticklabels(), rotation=0)

    ax_imp.set_ylabel("Coefficient", fontsize=9, color='white')
    ax_imp.tick_params(colors='white', labelsize=8)

    for spine in ax_imp.spines.values():
        spine.set_edgecolor("#FFBF00")

    plt.tight_layout()

    # ROC Curves
    # --- ROC Curve ---
    fig_roc, ax_roc = plt.subplots(figsize=(5, 3), dpi=120)
    fig_roc.patch.set_facecolor('#1E1E1E')
    ax_roc.set_facecolor('#1E1E1E')

    palette = sns.color_palette("colorblind", n_colors=len(model.classes_))
    for i, color in zip(range(len(model.classes_)), palette):
        fpr, tpr, _ = roc_curve(y_test == i, y_proba[:, i])
        auc_score = roc_auc_score(y_test == i, y_proba[:, i])
        ax_roc.plot(fpr, tpr, label=f'Class {i} (AUC = {auc_score:.2f})', color=color, linewidth=2)

    ax_roc.plot([0, 1], [0, 1], 'k--', lw=1, color='white', alpha=0.3)

    ax_roc.set_xlabel("False Positive Rate", fontsize=9, color='white')
    ax_roc.set_ylabel("True Positive Rate", fontsize=9, color='white')
    ax_roc.legend(facecolor='#1E1E1E', frameon=True, edgecolor='#FFBF00', labelcolor='white')
    ax_roc.tick_params(colors='white', labelsize=8)

    for spine in ax_roc.spines.values():
        spine.set_edgecolor("#FFBF00")

    # Add symbolic equation (Class 0)
    coefs = model.coef_[0]
    terms = [f"({coef:.2f} × {col})" for coef, col in zip(coefs, X.columns)]
    regression_eq = " + ".join(terms)
    metrics["equation"] = f"Category = {regression_eq}"

    return model, scaler, metrics, fig_cm, fig_imp, fig_roc
