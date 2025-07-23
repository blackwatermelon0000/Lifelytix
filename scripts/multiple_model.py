import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
import streamlit as st
import scipy.stats as stats

# Styled figure helper
def styled_fig():
    fig, ax = plt.subplots(figsize=(4.5, 3))
    fig.patch.set_facecolor('#1E1E1E')
    ax.set_facecolor('#1E1E1E')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor("#FFBF00")
    return fig, ax

def run_multiple_regression(df):
    # Inject CSS (only once)
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
        </style>
    """, unsafe_allow_html=True)

    # --- Data Prep ---
    df = df.dropna(subset=['Age', 'Weight', 'Height', 'Waist', 'Gender', 'BMI'])
    df['Gender_encoded'] = LabelEncoder().fit_transform(df['Gender'])

    X = df[['Age', 'Weight', 'Height', 'Waist', 'Gender_encoded']]
    y = df['BMI']

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)
    n, p = X.shape
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    intercept = model.intercept_
    equation = " + ".join([f"({coef:.2f} × {col})" for coef, col in zip(model.coef_, X.columns)])

    # --- 📈 Model Performance ---
    st.markdown("<h3 style='color: #FFBF00;'>📊 Model Performance</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='border: 1px solid #FFBF00; border-radius: 10px; background-color: #2A2A2A;
                    padding: 25px 15px; text-align: center; margin-bottom: 30px;'>
            <div style='display: flex; justify-content: space-around;'>
                <div><h4 style='color: #FFBF00;'>R² / Adj R²</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{r2:.2f} / {adj_r2:.2f}</span></div>
                <div><h4 style='color: #FFBF00;'>RMSE</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{rmse:.2f}</span></div>
                <div><h4 style='color: #FFBF00;'>MAE</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{mae:.2f}</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 🧮 Model Coefficients ---
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    coeff_html_rows = "".join([
        f"<tr><td>{row.Feature}</td><td>{row.Coefficient:.4f}</td></tr>"
        for _, row in coef_df.iterrows()
    ])

    st.markdown("""
        <div class='coeff-heading'>🧮 Model Coefficients</div>
        <div class="coeff-container">
            <table class="coeff-table">
                <thead><tr><th>Feature</th><th>Coefficient</th></tr></thead>
                <tbody>""" + coeff_html_rows + """</tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)

    # --- 📏 Regression Equation ---
    st.markdown(f"""
        <div style="margin-top:25px; border: 1px solid #FFBF00; border-radius: 10px;
                    padding: 15px; background-color: #1E1E1E;">
            <h4 style="color:#FFBF00; text-align:center; margin-bottom:10px;">📏 Regression Equation</h4>
            <div style="font-family: monospace; color: white; font-size: 20px; text-align: center;">
                BMI = {intercept:.2f} + {equation}
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 🧠 Multicollinearity Check (VIF) ---
    X_vif = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_vif.columns
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    # --- Styled VIF Table ---
    vif_html_rows = "".join([
        f"<tr><td>{row.Feature}</td><td>{row.VIF:.4f}</td></tr>"
        for _, row in vif_data.iterrows()
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
    st.markdown("<div class='coeff-heading'>📊 Regression Diagnostics</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = styled_fig()
        ax1.scatter(
            y, y_pred,
            color='#FBC02D',
            edgecolor='white',
            linewidth=0.25,
            alpha=0.4,  # slightly lower than 0.5
            s=10  # consistent with second plot
        )
        ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=1)
        ax1.set_title("Predicted vs Actual BMI", fontsize=12, color='#FFBF00')
        ax1.set_xlabel("Actual BMI", fontsize=9, color='white')
        ax1.set_ylabel("Predicted BMI", fontsize=9, color='white')
        ax1.tick_params(colors='white', labelsize=8)
        for spine in ax1.spines.values():
            spine.set_edgecolor("#FFBF00")
        st.pyplot(fig1, use_container_width=True)

    with col2:
        residuals = y - y_pred
        fig2, ax2 = styled_fig()
        ax2.scatter(y_pred, residuals, color='#FBC02D', edgecolor='white', alpha=0.4, s=10, linewidth=0.25)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
        ax2.set_title("Residuals vs Predicted", fontsize=12, color='#FFBF00')
        ax2.set_xlabel("Predicted BMI", fontsize=9, color='white')
        ax2.set_ylabel("Residuals", fontsize=9, color='white')
        ax2.tick_params(colors='white', labelsize=8)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#FFBF00")
        st.pyplot(fig2, use_container_width=True)

    # 📈 Residual Histogram
    fig3, ax3 = plt.subplots(figsize=(4.5, 3), dpi=120)
    fig3.patch.set_facecolor('#1E1E1E')
    ax3.set_facecolor('#1E1E1E')

    # Plot histogram with KDE
    sns.histplot(
        residuals,
        bins=40,
        kde=True,
        ax=ax3,
        color='#FBC02D',
        linewidth=0
    )

    # Optional: Horizontal mean reference line
    ax3.axvline(residuals.mean(), color='#00E5FF', linestyle='--', linewidth=1)

    # Titles and axis styling
    ax3.set_title("Histogram of Residuals", fontsize=12, color='#FFBF00')
    ax3.set_xlabel("Residual Value", fontsize=9, color='white')
    ax3.set_ylabel("Count", fontsize=9, color='white')
    ax3.tick_params(colors='white', labelsize=8)

    # Apply border styling
    for spine in ax3.spines.values():
        spine.set_edgecolor("#FFBF00")

    # Layout and display
    # Center-aligned layout for residual histogram
    center_col = st.columns([1, 2, 1])  # left spacing, center plot, right spacing
    with center_col[1]:
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)



