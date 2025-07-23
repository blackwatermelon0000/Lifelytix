import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from scripts.preprocess import load_and_preprocess_data
from scripts.linear_model import run_linear_regression
from scripts.multiple_model import run_multiple_regression
from scripts.logistic_model import train_logistic_model
from scripts.decision_tree import run_decision_tree
from scripts.naive_bayes import run_naive_bayes
from scripts.neural_network import run_neural_network
from scripts.EDA import run_all_eda_plots

st.set_page_config(layout="wide", page_title="Lifelytix: AI Health Analyzer")
st.markdown("""
<style>
    /* Typography and core styles */
    body, html {
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #FFBF00 !important;
        font-family: 'Segoe UI', sans-serif;
        text-align: center;
    }

    /* Table/Charts scroll fix */
    .stPlotlyChart, .stImage, .stTable, .stDataFrame {
        max-width: 100%;
        overflow-x: auto;
    }
    /* Customize tab labels */
.css-10trblm {  /* Text inside each tab */
    font-weight: bold;
    color: white;
}

/* Highlight selected tab with amber */
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    border-bottom: 3px solid transparent;
    color: white;
    font-weight: bold;
}

.stTabs [aria-selected="true"] {
    border-bottom: 3px solid #FFBF00;
    color: #FFBF00;
}
.stTabs [aria-selected="false"] {
    opacity: 0.6;
}


</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.custom-table {
    border-collapse: collapse;
    width: 100%;
    border: 2px solid #FFBF00;
    border-radius: 10px;
    overflow: hidden;
    font-size: 14px;
    background-color: #1E1E1E;
    color: white;
}
.custom-table th {
    background-color: #1E1E1E;
    color: #FFBF00;
    border: 1px solid #FFBF00;
    padding: 10px;
    text-align: center;
}
.custom-table td {
    border: 1px solid #FFBF00;
    padding: 10px;
    text-align: center;
}
.custom-table tr:nth-child(even) {
    background-color: #2A2A2A;
}
.custom-table tr:nth-child(odd) {
    background-color: #1E1E1E;
}
</style>
""", unsafe_allow_html=True)

# Load Data
with st.spinner("Loading Lifelytix ⏳"):
    df, X_train, X_test, y_train, y_test = load_and_preprocess_data()
    log_model, log_scaler, log_metrics, log_fig_cm, log_fig_imp, log_fig_roc = train_logistic_model(df)

    st.markdown("""
        <div style='background-color:#1E1E1E;padding:20px;border-radius:10px;margin-bottom:20px'>
            <h2 style='color:#FFBF00;text-align:center;'>Welcome to Lifelytix</h2>
            <p style='color:white;text-align:center;'>Use AI-powered insights to explore your health data, trends, and predictions.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align: center; margin-top: 10px; font-size: 14px;'>
        <span style='color:#FFBF00;'>📊 Data Source:</span>
        <a style='color:#1E90FF; text-decoration: none;' href='https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015' target='_blank'>
            NHANES 2015–2016
        </a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

st.markdown("""
    <style>
    .coeff-heading {
        color: #FFBF00;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin: 30px 0 15px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Create Tabs
tabs = st.tabs([
    "🏠 Home", "📈 EDA", "📉 Regression",
    "🔍 Classification", "🧠 Neural Net", "🧪 Try It Yourself"
])
# --- 🏠 HOME TAB ---
with tabs[0]:
    # Inject custom styles
    st.markdown("""
    <style>
        .amber-label {
            color: #FFBF00;
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 5px;
            display: block;
        }

        div[data-baseweb="select"] > div {
            border: 2px solid #FFBF00 !important;
            border-radius: 8px !important;
            background-color: #1E1E1E !important;
        }

        .stat-box {
            border: 2px solid #FFBF00;
            border-radius: 10px;
            padding: 10px 0;
            text-align: center;
            background-color: #1E1E1E;
            color: white;
        }
        .stat-box .label {
            font-size: 14px;
            color: #FFBF00;
            font-weight: bold;
        }
        .stat-box .value {
            font-size: 28px;
            font-weight: bold;
        }

        .custom-table {
            border-collapse: collapse;
            width: 100%;
            border: 2px solid #FFBF00;
            border-radius: 10px;
            overflow: hidden;
            font-size: 14px;
            background-color: #1E1E1E;
            color: white;
        }
        .custom-table th {
            background-color: #1E1E1E;
            color: #FFBF00;
            border: 1px solid #FFBF00;
            padding: 10px;
            text-align: center;
        }
        .custom-table td {
            border: 1px solid #FFBF00;
            padding: 10px;
            text-align: center;
        }
        .custom-table tr:nth-child(even) {
            background-color: #2A2A2A;
        }
        .custom-table tr:nth-child(odd) {
            background-color: #1E1E1E;
        }
    </style>
    """, unsafe_allow_html=True)

    # Section Header
    st.markdown("<h2 style='color: #FFBF00; font-weight: bold; text-align: center;'>🎛️ Data Statistics</h2>", unsafe_allow_html=True)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<span class='amber-label'>Select Gender</span>", unsafe_allow_html=True)
        gender_filter = st.selectbox("", options=["All"] + sorted(df["Gender"].dropna().unique()))

    with col2:
        st.markdown("<span class='amber-label'>Age Range</span>", unsafe_allow_html=True)
        age_group = st.slider(
            label="",
            min_value=0,
            max_value=100,
            value=(20, 60),
            key="age_slider"
        )

    # Filter dataset
    filtered_df = df.copy()
    if gender_filter != "All":
        filtered_df = filtered_df[filtered_df["Gender"] == gender_filter]
    filtered_df = filtered_df[(filtered_df["Age"] >= age_group[0]) & (filtered_df["Age"] <= age_group[1])]

    # Stats Display
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
            <div class='stat-box'>
                <div class='label'>Avg. BMI</div>
                <div class='value'>{filtered_df['BMI'].mean():.1f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class='stat-box'>
                <div class='label'>Avg. Age</div>
                <div class='value'>{filtered_df['Age'].mean():.0f} yrs</div>
            </div>
        """, unsafe_allow_html=True)

    if gender_filter == "All":
        total = df["Gender"].value_counts(normalize=True)
        male_pct = total.get("Male", 0) * 100
        female_pct = total.get("Female", 0) * 100
    else:
        male_pct = female_pct = None
        if gender_filter == "Male":
            male_pct = 100
        elif gender_filter == "Female":
            female_pct = 100

    with col3:
        st.markdown(f"""
            <div class='stat-box'>
                <div class='label'>% Male</div>
                <div class='value'>{male_pct if male_pct is not None else 0:.0f}%</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class='stat-box'>
                <div class='label'>% Female</div>
                <div class='value'>{female_pct if female_pct is not None else 0:.0f}%</div>
            </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
            <div class='stat-box'>
                <div class='label'>Sys BP Avg</div>
                <div class='value'>{filtered_df['Systolic BP'].mean():.0f}</div>
            </div>
        """, unsafe_allow_html=True)

    # Add spacing before table
    st.markdown("<div style='height:40px;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h2 style='color: #FFBF00; font-weight: bold;'>📋 Filtered Sample of Dataset</h2>", unsafe_allow_html=True)

    # Prepare and render table
    styled_df = filtered_df[['Age', 'Gender', 'Weight', 'Height', 'Waist', 'BMI']].copy()
    styled_df = styled_df.sort_values(by='Age')
    styled_df['Age'] = styled_df['Age'].astype(int)
    styled_df['Weight'] = styled_df['Weight'].map('{:.2f}'.format)
    styled_df['Height'] = styled_df['Height'].map('{:.2f}'.format)
    styled_df['Waist'] = styled_df['Waist'].map('{:.2f}'.format)
    styled_df['BMI'] = styled_df['BMI'].map('{:.2f}'.format)

    html_table = styled_df.to_html(index=False, classes="custom-table", border=0, justify="center")
    st.markdown(html_table, unsafe_allow_html=True)

# --- 📈 EDA TAB ---
with tabs[1]:
    st.markdown("""
    <style>
        .section {
            padding: 15px;
        }
        .section h2 {
            color: #FFBF00;
            text-align: center;
            margin-bottom: 5px;
            font-size: 26px;
        }
        .section p {
            text-align: center;
            color: white;
            font-size: 14px;
            margin-top: 0;
        }
    </style>
    <div class="section">
        <h2>🔬 Exploratory Data Analysis</h2>
        <p>Explore patterns and trends in health metrics using visual tools.</p>
    </div>
    """, unsafe_allow_html=True)

    run_all_eda_plots(df)
# --- 📉 REGRESSION TAB ---
with tabs[2]:
    # Top heading
    st.markdown("""
        <div style='text-align: center; margin-top: 20px; margin-bottom: 30px;'>
            <h1 style='color: #FFBF00; font-size: 36px;'>📉 Regression Models</h1>
            <p style='color: #cccccc; font-size: 16px;'>Explore predictive relationships using regression techniques.</p>
        </div>
    """, unsafe_allow_html=True)

    sub_tabs = st.tabs([
        "🧮 Simple Linear Regression",
        "📊 Multiple Linear Regression",
        "🧪 Logistic Regression"
    ])

    # --- Simple Linear ---
    with sub_tabs[0]:
        run_linear_regression(df)

    # --- Multiple Linear ---
    with sub_tabs[1]:
        run_multiple_regression(df)

    # --- Logistic Regression ---
    with sub_tabs[2]:
        st.markdown("""
            <h3 style='color: #FFBF00;'>📊 Model Performance</h3>
            <div style='border: 1px solid #FFBF00; border-radius: 10px; background-color: #2A2A2A;
                        padding: 25px 15px; text-align: center; margin-bottom: 30px;'>
                <div style='display: flex; justify-content: space-around;'>
                    <div><h4 style='color: #FFBF00;'>Accuracy</h4>
                        <span style='color: white; font-size: 22px; font-weight: bold;'>{:.2f}</span></div>
                    <div><h4 style='color: #FFBF00;'>Precision</h4>
                        <span style='color: white; font-size: 22px; font-weight: bold;'>{:.2f}</span></div>
                    <div><h4 style='color: #FFBF00;'>Recall</h4>
                        <span style='color: white; font-size: 22px; font-weight: bold;'>{:.2f}</span></div>
                    <div><h4 style='color: #FFBF00;'>ROC AUC</h4>
                        <span style='color: white; font-size: 22px; font-weight: bold;'>{:.2f}</span></div>
                </div>
            </div>
        """.format(
            log_metrics['accuracy'],
            log_metrics['precision'],
            log_metrics['recall'],
            log_metrics['roc_auc']
        ), unsafe_allow_html=True)

        # Equation
        if "equation" in log_metrics:
            st.markdown(f"""
                <div style="margin-top:25px; border: 1px solid #FFBF00; border-radius: 10px;
                            padding: 15px; background-color: #1E1E1E;">
                    <h4 style="color:#FFBF00; text-align:center; margin-bottom:10px;">🧮 Logistic Model Equation (Class 0)</h4>
                    <div style="font-family: monospace; color: white; font-size: 18px; text-align: center;">
                        {log_metrics["equation"]}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Diagnostics
        st.markdown("<div class='coeff-heading'>📊 Logistic Model Diagnostics</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔍 Confusion Matrix")
            st.pyplot(log_fig_cm, use_container_width=True)

        with col2:
            st.markdown("#### 📈 ROC Curve")
            st.pyplot(log_fig_roc, use_container_width=True)

        center = st.columns([1, 2, 1])
        with center[1]:
            st.markdown("#### 📊 Feature Importance")
            st.pyplot(log_fig_imp, use_container_width=True)

# --- 🔍 CLASSIFICATION TAB ---
with tabs[3]:
    st.markdown("""
        <div style='text-align: center; margin-top: 20px; margin-bottom: 30px;'>
            <h1 style='color: #FFBF00; font-size: 36px;'>🔍 Classification Models</h1>
            <p style='color: #cccccc; font-size: 16px;'>Evaluate machine learning classifiers on health indicators.</p>
        </div>
    """, unsafe_allow_html=True)

    class_subtabs = st.tabs([
        "🌳 Decision Tree",
        "🧮 Naive Bayes"
    ])
    # 🌳 Decision Tree
    with class_subtabs[0]:
        _ = run_decision_tree(df)  # Suppress return value

    # 🧮 Naive Bayes
    with class_subtabs[1]:
        _ = run_naive_bayes(df)  # Suppress return value and avoid extra output

# --- 🧠 NEURAL NET TAB ---
with tabs[4]:
    st.markdown('<div class="section">', unsafe_allow_html=True)
    model, metrics, fig_cm = run_neural_network(df)

    # --- 1️⃣ Model Performance ---
    st.markdown("<h3 style='color: #FFBF00;'>📊 Model Performance</h3>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style='border: 1px solid #FFBF00; border-radius: 10px; background-color: #2A2A2A;
                    padding: 25px 15px; text-align: center; margin-bottom: 30px;'>
            <div style='display: flex; justify-content: space-around; flex-wrap: wrap;'>
                <div><h4 style='color: #FFBF00;'>Train Accuracy</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{metrics['accuracy_train']:.2f}</span></div>
                <div><h4 style='color: #FFBF00;'>Test Accuracy</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{metrics['accuracy_test']:.2f}</span></div>
                <div><h4 style='color: #FFBF00;'>Precision</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{metrics['precision']:.2f}</span></div>
                <div><h4 style='color: #FFBF00;'>Recall</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{metrics['recall']:.2f}</span></div>
                <div><h4 style='color: #FFBF00;'>F1-Score</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{metrics['f1-score']:.2f}</span></div>
                <div><h4 style='color: #FFBF00;'>Log Loss</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{metrics['log_loss']:.2f}</span></div>
                <div><h4 style='color: #FFBF00;'>ROC AUC</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{metrics['auc_roc']:.2f}</span></div>
                <div><h4 style='color: #FFBF00;'>CV Accuracy</h4>
                    <span style='color: white; font-size: 22px; font-weight: bold;'>{metrics['cross_val_mean']:.2f} ± {metrics['cross_val_std']:.2f}</span></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- 2️⃣ Styled Prediction Probabilities (Top 5) ---
    st.markdown("""
        <h3 style='color:#FFBF00; font-size:26px; text-align:center;'>🔢 Prediction Probabilities</h3>
        <style>
            .coeff-container {{
                border: 1px solid #FFBF00;
                border-radius: 10px;
                background-color: #2A2A2A;
                padding: 25px 30px;
                margin-top: 15px;
                margin-bottom: 30px;
            }}
            .coeff-table {{
                width: 100%;
                border-collapse: collapse;
                font-family: monospace;
            }}
            .coeff-table th {{
                color: #FFBF00;
                text-align: center;
                padding: 10px;
                font-size: 17px;
                border-bottom: 1px solid #444;
            }}
            .coeff-table td {{
                padding: 10px;
                font-size: 16px;
                text-align: center;
                border-bottom: 1px solid #444;
                color: white;
            }}
        </style>
    """, unsafe_allow_html=True)

    prob_df = pd.DataFrame(model.predict_proba(metrics['X_test']),
                           columns=[f"Class {i}" for i in model.classes_]).head()

    prob_html_rows = "".join([
        "<tr>" + "".join(f"<td>{val:.4f}</td>" for val in row) + "</tr>"
        for _, row in prob_df.iterrows()
    ])

    st.markdown(f"""
        <div class="coeff-container">
            <table class="coeff-table">
                <thead>
                    <tr>{''.join(f"<th>{col}</th>" for col in prob_df.columns)}</tr>
                </thead>
                <tbody>
                    {prob_html_rows}
                </tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)

    # --- 3️⃣ Classification Report ---
    st.markdown("<h3 style='color:#FFBF00; font-size:26px; text-align:center;'>📋 Classification Report</h3>",
                unsafe_allow_html=True)
    report_df = pd.DataFrame(metrics['report']).T.round(4)
    st.dataframe(report_df.style.background_gradient(cmap='coolwarm'), use_container_width=True)

    # --- 4️⃣ Confusion Matrix & Loss Curve ---
    st.markdown("<h3 style='color:#FFBF00; font-size:26px; text-align:center;'>🔍 Confusion Matrix & 📉 Loss Curve</h3>",
                unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
    with col_left:
        st.pyplot(fig_cm, use_container_width=True)

    with col_right:
        if hasattr(model, 'loss_curve_'):
            fig_loss, ax_loss = plt.subplots(figsize=(4.5, 3))
            fig_loss.patch.set_facecolor('#1E1E1E')
            ax_loss.set_facecolor('#1E1E1E')
            ax_loss.plot(model.loss_curve_, label='Loss', color='#FFBF00')
            ax_loss.set_title("Loss Over Epochs", fontsize=11, color='#FFBF00')
            ax_loss.set_xlabel("Epoch", fontsize=9, color='white')
            ax_loss.set_ylabel("Loss", fontsize=9, color='white')
            ax_loss.tick_params(colors='white', labelsize=8)
            for spine in ax_loss.spines.values():
                spine.set_edgecolor("#FFBF00")
            ax_loss.legend()
            st.pyplot(fig_loss, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
with tabs[5]:  # Make sure this is the correct tab index
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:#FFBF00; text-align:center;'>🧪 BMI </h2>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; color:#cccccc;'>Navigate through each step to explore your health with AI models.</p>",
        unsafe_allow_html=True)
    st.markdown("### 🔢 Live BMI Calculator")
    col1, col2 = st.columns(2)
    with col1:
        weight = st.number_input("Enter Weight (kg)", min_value=20.0, max_value=200.0, value=65.0)
    with col2:
        height_cm = st.number_input("Enter Height (cm)", min_value=100.0, max_value=250.0, value=170.0)

    height_m = height_cm / 100
    bmi = round(weight / (height_m ** 2), 2)
    st.session_state["bmi"] = bmi
    st.info(f"🧮 Your Calculated BMI: **{bmi}**")

    st.markdown("</div>", unsafe_allow_html=True)