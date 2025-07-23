def run_naive_bayes(df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder, label_binarize
    import streamlit as st

    # --- Preprocessing ---
    df = df.dropna(subset=['BMI', 'Age', 'Waist', 'Gender'])
    df['BMICategory'] = pd.qcut(df['BMI'], q=4, labels=[0, 1, 2, 3])
    df['Gender_encoded'] = LabelEncoder().fit_transform(df['Gender'])

    X = df[['Age', 'Waist', 'Gender_encoded']]
    y = df['BMICategory'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    y_train_pred = model.predict(X_train)

    acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    logloss = log_loss(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc_ovr = roc_auc_score(label_binarize(y_test, classes=[0, 1, 2, 3]), y_proba, multi_class='ovr')
    cross_val = cross_val_score(model, X, y, cv=5)

    # --- 📊 Model Performance ---
    st.markdown("<h3 style='color:#FFBF00; font-size:26px; text-align:center; margin:30px 0;'>📊 Model Performance</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='border: 1px solid #FFBF00; border-radius: 10px; background-color: #2A2A2A;
                    padding: 25px 15px; text-align: center; margin-bottom: 30px;'>
            <div style='display: flex; justify-content: space-around;'>
                <div><h4 style='color: #FFBF00;'>Accuracy</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{acc:.2f}</span></div>
                <div><h4 style='color: #FFBF00;'>Precision</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{report['weighted avg']['precision']:.2f}</span></div>
                <div><h4 style='color: #FFBF00;'>Recall</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{report['weighted avg']['recall']:.2f}</span></div>
                <div><h4 style='color: #FFBF00;'>ROC AUC</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{auc_ovr:.2f}</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 🧮 Model Summary ---
    st.markdown("<h3 style='color:#FFBF00; font-size:26px; text-align:center; margin:30px 0;'>🧮 Model Summary</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="coeff-container">
            <table class="coeff-table">
                <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                <tbody>
                    <tr><td>Training Accuracy</td><td>{train_acc:.2f}</td></tr>
                    <tr><td>Log Loss</td><td>{logloss:.2f}</td></tr>
                    <tr><td>CV Accuracy (Mean ± Std)</td><td>{cross_val.mean():.2f} ± {cross_val.std():.2f}</td></tr>
                </tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)

    # --- 📊 Prior Class Probabilities ---
    st.markdown("<h3 style='color:#FFBF00; font-size:26px; text-align:center; margin:30px 0;'>📊 Prior Class Probabilities</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div class="coeff-container">
            <table class="coeff-table">
                <thead><tr><th>Class</th><th>Prior Probability</th></tr></thead>
                <tbody>
                    {''.join(f"<tr><td>{int(cls)}</td><td>{prob:.2f}</td></tr>" for cls, prob in zip(model.classes_, model.class_prior_))}
                </tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)

    # --- 🔢 Prediction Probabilities ---
    st.markdown("<h3 style='color:#FFBF00; font-size:26px; text-align:center; margin:30px 0;'>🔢 Prediction Probabilities</h3>", unsafe_allow_html=True)
    sample_html_rows = ""
    for idx, row in enumerate(y_proba[:5]):
        row_html = "".join([f"<td>{prob:.2f}</td>" for prob in row])
        sample_html_rows += f"<tr><td>{idx+1}</td>{row_html}</tr>"

    st.markdown(f"""
        <div class="coeff-container">
            <table class="coeff-table">
                <thead>
                    <tr>
                        <th>Sample</th>
                        {''.join(f'<th>Class {int(c)}</th>' for c in model.classes_)}
                    </tr>
                </thead>
                <tbody>
                    {sample_html_rows}
                </tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)

    # --- 📋 Classification Report ---
    st.markdown("<h3 style='color:#FFBF00; font-size:26px; text-align:center;'>📋 Classification Report</h3>", unsafe_allow_html=True)
    # Round numeric values to 4 decimals and reformat all float cells
    report_df = pd.DataFrame(report).T
    report_df = report_df.round(4)  # ✅ Ensure only 4 decimal places

    # Optional: convert to styled format with gradient
    styled_report = report_df.style.format(precision=4).background_gradient(cmap='coolwarm', axis=0)
    st.dataframe(styled_report, use_container_width=True)
    # --- 🔍 Confusion Matrix ---
    st.markdown("<h3 style='color:#FFBF00; font-size:26px; text-align:center;'>🔍 Confusion Matrix</h3>",
                unsafe_allow_html=True)

    # Create compact confusion matrix figure
    fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
    fig_cm.patch.set_facecolor('#1E1E1E')
    ax_cm.set_facecolor('#1E1E1E')

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='coolwarm',
        cbar=True,
        square=True,
        ax=ax_cm,
        annot_kws={"size": 9, "color": "black"},
        linewidths=0.5,
        linecolor='black'
    )
    ax_cm.set_xlabel("Predicted", fontsize=9, color='white')
    ax_cm.set_ylabel("Actual", fontsize=9, color='white')
    ax_cm.tick_params(colors='white', labelsize=8)
    ax_cm.set_title("Confusion Matrix", fontsize=11, color='#FFBF00')

    for spine in ax_cm.spines.values():
        spine.set_edgecolor("#FFBF00")

    # ✅ Use Streamlit column layout to center
    center_col = st.columns([1, 2, 1])
    with center_col[1]:
        plt.tight_layout()
        st.pyplot(fig_cm, use_container_width=True)

    # --- Return ---
    metrics = {
        "accuracy_test": acc,
        "accuracy_train": train_acc,
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1-score": report['weighted avg']['f1-score'],
        "auc_roc": auc_ovr,
        "log_loss": logloss,
        "cross_val_mean": cross_val.mean(),
        "cross_val_std": cross_val.std()
    }
    return model, metrics, fig_cm
