import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, label_binarize
import streamlit as st
import streamlit.components.v1 as components
import pydotplus
import base64
from io import StringIO

def run_decision_tree(df):
    df = df.dropna(subset=['BMI', 'Age', 'Waist', 'Gender'])
    df['BMICategory'] = pd.qcut(df['BMI'], q=4, labels=[0, 1, 2, 3])
    df['Gender_encoded'] = LabelEncoder().fit_transform(df['Gender'])

    X = df[['Age', 'Waist', 'Gender_encoded']]
    y = df['BMICategory'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    auc_ovr = roc_auc_score(label_binarize(y_test, classes=[0, 1, 2, 3]), model.predict_proba(X_test), multi_class='ovr')
    cross_val = cross_val_score(model, X, y, cv=5)

    # 📊 Model Performance
    st.markdown(f"""
        <h3 style='color: #FFBF00;'>📊 Model Performance</h3>
        <div style='border: 1px solid #FFBF00; border-radius: 10px; background-color: #2A2A2A;
                    padding: 25px 15px; text-align: center; margin-bottom: 30px;'>
            <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
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

    # --- 🧮 Tree Model Summary ---
    st.markdown(
        "<h3 style='color:#FFBF00; font-size:26px; text-align:center; margin:30px 0;'>🧮 Tree Model Summary</h3>",
        unsafe_allow_html=True)
    st.markdown(f"""
            <div class="coeff-container">
                <table class="coeff-table">
                    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                    <tbody>
                        <tr><td>Training Accuracy</td><td>{train_acc:.2f}</td></tr>
                        <tr><td>Tree Depth</td><td>{model.get_depth()}</td></tr>
                        <tr><td>Leaves</td><td>{model.get_n_leaves()}</td></tr>
                        <tr><td>CV Accuracy (Mean ± Std)</td><td>{cross_val.mean():.2f} ± {cross_val.std():.2f}</td></tr>
                    </tbody>
                </table>
            </div>
        """, unsafe_allow_html=True)

    # --- 📊 Model Diagnostics ---
    st.markdown(
        "<h3 style='color:#FFBF00; font-size:26px; text-align:center; margin:30px 0;'>📊 Model Diagnostics</h3>",
        unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔍 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        fig_cm.patch.set_facecolor('#1E1E1E')
        ax.set_facecolor('#1E1E1E')
        sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', cbar=True, square=True, ax=ax,
                    annot_kws={"size": 10, "color": "black"})
        ax.set_xlabel("Predicted", color='white')
        ax.set_ylabel("Actual", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor("#FFBF00")
        st.pyplot(fig_cm)

    with col2:
        st.markdown("### 📈 Feature Importance")
        feature_imp = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        fig_imp, ax = plt.subplots()
        fig_imp.patch.set_facecolor('#1E1E1E')
        ax.set_facecolor('#1E1E1E')
        sns.barplot(x='Feature', y='Importance', data=feature_imp,
                    palette=sns.color_palette("colorblind", n_colors=len(feature_imp)))
        ax.set_xlabel("Feature", color='white')
        ax.set_ylabel("Importance", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor("#FFBF00")
        st.pyplot(fig_imp)

    # 🌳 Zoomable Decision Tree
    st.markdown("### 🌳 Zoomable Decision Tree", unsafe_allow_html=True)
    dot_data = StringIO()
    export_graphviz(
        model,
        out_file=dot_data,
        feature_names=X.columns,
        class_names=["0", "1", "2", "3"],
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    png_image = graph.create_png()
    encoded = base64.b64encode(png_image).decode("utf-8")
    html = f"""
        <div style="border: 1px solid #FFBF00; border-radius: 10px; background: #1E1E1E; overflow: auto;">
            <div id="zoom-container" style="transform: scale(1); transform-origin: 0 0;">
                <img src="data:image/png;base64,{encoded}" style="width: 100%;" />
            </div>
        </div>
        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                const container = document.getElementById('zoom-container');
                let scale = 1;
                container.addEventListener("wheel", function(e) {{
                    if (e.ctrlKey) {{
                        e.preventDefault();
                        scale += e.deltaY * -0.001;
                        scale = Math.min(Math.max(0.5, scale), 3);
                        container.style.transform = `scale(${{scale}})`;
                    }}
                }}, {{ passive: false }});
            }});
        </script>
    """
    components.html(html, height=1000, scrolling=True)

    # Return for inference use
    metrics = {
        "accuracy_test": acc,
        "accuracy_train": train_acc,
        "precision": report['weighted avg']['precision'],
        "recall": report['weighted avg']['recall'],
        "f1-score": report['weighted avg']['f1-score'],
        "auc_roc": auc_ovr,
        "cross_val_mean": cross_val.mean(),
        "cross_val_std": cross_val.std(),
        "tree_depth": model.get_depth(),
        "num_leaves": model.get_n_leaves()
    }

    return model, metrics, fig_cm
