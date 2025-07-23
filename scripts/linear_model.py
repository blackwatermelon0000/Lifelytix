import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st
import scipy.stats as stats


# Styled figure helper (consistent with multiple regression)
def styled_fig():
    fig, ax = plt.subplots(figsize=(4.5, 3))
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor("#FFBF00")
    return fig, ax


def run_linear_regression(df):
    # Inject CSS (consistent styling)
    st.markdown("""
        <style>
        .coeff-container {
            border: 1px solid #FFBF00;
            border-radius: 10px;
            background-color: #2A2A2A;
            padding: 25px 30px;
            margin-top: 15px;
            margin-bottom: 30px;
        }
        .coeff-heading {
            color: #FFBF00;
            text-align: center;
            margin: 35px 0 10px 0;
            font-size: 30px;
            font-weight: bold;
        }
        .coeff-table {
            width: 100%;
            border-collapse: collapse;
            font-family: monospace;
        }
        .coeff-table th {
            color: #FFBF00;
            text-align: left;
            padding: 10px;
            font-size: 18px;
            border-bottom: 1px solid #444;
        }
        .coeff-table td {
            padding: 10px;
            font-size: 17px;
            border-bottom: 1px solid #444;
        }
        .coeff-table td:first-child {
            color: #FFBF00;
            font-weight: bold;
        }
        .coeff-table td:last-child {
            text-align: right;
            color: white;
        }
        .metric-box {
            border: 1px solid #FFBF00;
            border-radius: 10px;
            background-color: #2A2A2A;
            padding: 15px;
            margin: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Check for required columns
    required_cols = ['BMI', 'Systolic BP', 'Age', 'Gender_encoded', 'Waist']
    if not all(col in df.columns for col in required_cols):
        st.error("❌ Required columns not found in dataset")
        return None, {}, None, None, None

    # Data preparation
    df = df.dropna(subset=required_cols)
    if len(df) == 0:
        st.error("❌ No valid rows after dropping missing values")
        return None, {}, None, None, None

    # Encode gender if not already encoded
    if 'Gender' in df.columns and 'Gender_encoded' not in df.columns:
        df['Gender_encoded'] = LabelEncoder().fit_transform(df['Gender'])

    # --- Simple Regression ---
    X_simple = df[['BMI']]
    y = df['Systolic BP']
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y, test_size=0.2, random_state=42)
    model_s = LinearRegression().fit(X_train_s, y_train_s)
    y_pred_s = model_s.predict(X_test_s)

    # --- Extended Regression ---
    X_ext = df[['BMI', 'Age', 'Gender_encoded', 'Waist']]
    X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(X_ext, y, test_size=0.2, random_state=42)
    model_e = LinearRegression().fit(X_train_e, y_train_e)
    y_pred_e = model_e.predict(X_test_e)
    residuals_e = y_test_e - y_pred_e

    # --- Metrics Calculation ---
    metrics = {
        'simple': {
            'mse': mean_squared_error(y_test_s, y_pred_s),
            'rmse': np.sqrt(mean_squared_error(y_test_s, y_pred_s)),
            'mae': mean_absolute_error(y_test_s, y_pred_s),
            'r2': r2_score(y_test_s, y_pred_s)
        },
        'extended': {
            'mse': mean_squared_error(y_test_e, y_pred_e),
            'rmse': np.sqrt(mean_squared_error(y_test_e, y_pred_e)),
            'mae': mean_absolute_error(y_test_e, y_pred_e),
            'r2': r2_score(y_test_e, y_pred_e)
        }
    }

    # --- 🎯 Model Performance Display ---
    st.markdown("<div class='coeff-heading'>📈 Systolic BP Prediction Models</div>", unsafe_allow_html=True)

    # Model comparison metrics
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class='metric-box'>
                <h4 style='color: #FFBF00; text-align: center;'>Simple Model (BMI Only)</h4>
                <table style='width: 100%; color: white;'>
                    <tr><td>R²</td><td style='text-align: right;'>{:.3f}</td></tr>
                    <tr><td>RMSE</td><td style='text-align: right;'>{:.2f}</td></tr>
                    <tr><td>MAE</td><td style='text-align: right;'>{:.2f}</td></tr>
                </table>
            </div>
        """.format(
            metrics['simple']['r2'],
            metrics['simple']['rmse'],
            metrics['simple']['mae']
        ), unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class='metric-box'>
                <h4 style='color: #FFBF00; text-align: center;'>Extended Model</h4>
                <table style='width: 100%; color: white;'>
                    <tr><td>R²</td><td style='text-align: right;'>{:.3f}</td></tr>
                    <tr><td>RMSE</td><td style='text-align: right;'>{:.2f}</td></tr>
                    <tr><td>MAE</td><td style='text-align: right;'>{:.2f}</td></tr>
                </table>
            </div>
        """.format(
            metrics['extended']['r2'],
            metrics['extended']['rmse'],
            metrics['extended']['mae']
        ), unsafe_allow_html=True)

    # --- 🧮 Extended Model Coefficients ---
    coef_df = pd.DataFrame({
        'Feature': X_ext.columns,
        'Coefficient': model_e.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    coeff_html_rows = "".join([
        f"<tr><td>{row.Feature}</td><td>{row.Coefficient:.4f}</td></tr>"
        for _, row in coef_df.iterrows()
    ])

    st.markdown("""
        <div class='coeff-heading'>🧮 Extended Model Coefficients</div>
        <div class="coeff-container">
            <table class="coeff-table">
                <thead><tr><th>Feature</th><th>Coefficient</th></tr></thead>
                <tbody>""" + coeff_html_rows + """</tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)

    # --- 📏 Regression Equation ---
    equation = " + ".join([f"({coef:.2f} × {col})" for coef, col in zip(model_e.coef_, X_ext.columns)])
    st.markdown(f"""
        <div style="margin-top:25px; border: 1px solid #FFBF00; border-radius: 10px;
                    padding: 15px; background-color: #1E1E1E;">
            <h4 style="color:#FFBF00; text-align:center; margin-bottom:10px;">📏 Regression Equation</h4>
            <div style="font-family: monospace; color: white; font-size: 20px; text-align: center;">
                Systolic BP = {model_e.intercept_:.2f} + {equation}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 🧠 Multicollinearity Check (VIF) ---
    X_vif = sm.add_constant(X_ext)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]

    vif_html_rows = "".join([
        f"<tr><td>{row.Feature}</td><td>{row.VIF:.2f}</td></tr>"
        for _, row in vif_data.iterrows() if row.Feature != 'const'
    ])

    st.markdown("""
        <div class='coeff-heading'>🧠 Multicollinearity Check (VIF)</div>
        <div class="coeff-container">
            <table class="coeff-table">
                <thead><tr><th>Feature</th><th>VIF</th></tr></thead>
                <tbody>""" + vif_html_rows + """</tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)

    # --- 📊 Regression Diagnostics ---
    st.markdown("<div class='coeff-heading'>📊 Regression Diagnostics</div>", unsafe_allow_html=True)

    # Create figures
    fig1, ax1 = styled_fig()
    ax1.scatter(X_test_s['BMI'], y_test_s, color='#FBC02D', edgecolor='white', alpha=0.4, s=10, linewidth=0.25)
    ax1.plot(X_test_s['BMI'], y_pred_s, color='#00E5FF', linewidth=2)
    ax1.set_title("Simple Linear Regression", fontsize=12, color='#FFBF00')
    ax1.set_xlabel("BMI", fontsize=9, color='white')
    ax1.set_ylabel("Systolic BP", fontsize=9, color='white')

    fig2, ax2 = styled_fig()
    ax2.scatter(y_test_e, y_pred_e, color='#FBC02D', edgecolor='white', alpha=0.4, s=10, linewidth=0.25)
    ax2.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1)
    ax2.set_title("Extended Model Predictions", fontsize=12, color='#FFBF00')
    ax2.set_xlabel("Actual Systolic BP", fontsize=9, color='white')
    ax2.set_ylabel("Predicted Systolic BP", fontsize=9, color='white')

    fig3, ax3 = plt.subplots(figsize=(4.5, 3), dpi=120)
    fig3.patch.set_facecolor('#1E1E1E')
    ax3.set_facecolor('#1E1E1E')
    sns.histplot(residuals_e, bins=40, kde=True, ax=ax3, color='#FBC02D', linewidth=0)
    ax3.axvline(residuals_e.mean(), color='#00E5FF', linestyle='--', linewidth=1)
    ax3.set_title("Residuals Distribution", fontsize=12, color='#FFBF00')
    ax3.set_xlabel("Residual Value", fontsize=9, color='white')
    ax3.set_ylabel("Count", fontsize=9, color='white')
    ax3.tick_params(colors='white', labelsize=8)
    for spine in ax3.spines.values():
        spine.set_edgecolor("#FFBF00")

    # Display figures in a grid
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig1, use_container_width=True)
    with col2:
        st.pyplot(fig2, use_container_width=True)

    center_col = st.columns([1, 2, 1])
    with center_col[1]:
        st.pyplot(fig3, use_container_width=True)

    return model_e, metrics, fig1, fig2, fig3