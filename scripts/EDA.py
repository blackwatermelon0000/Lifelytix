import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# 🔧 Global Plot Style
plt.style.use('dark_background')
plt.rcParams.update({
    'figure.max_open_warning': 0,
    'axes.titlesize': 12,
    'axes.titlecolor': '#FFBF00',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.edgecolor': '#FFBF00',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'figure.facecolor': '#1E1E1E',
    'axes.facecolor': '#1E1E1E'
})

def styled_fig():
    fig, ax = plt.subplots(figsize=(8, 4))
    return fig, ax
def plot_bmi_hist(df):
    fig, ax = styled_fig()
    sns.histplot(df['BMI'].dropna(), bins=30, kde=True, ax=ax, color='#FFBF00')
    ax.set_title("BMI Distribution")
    ax.set_xlabel("BMI")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig

def plot_weight_hist(df):
    fig, ax = styled_fig()
    sns.histplot(df['Weight'].dropna(), bins=30, kde=True, ax=ax, color='#FFBF00')
    ax.set_title("Weight Distribution")
    ax.set_xlabel("Weight (kg)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig

def plot_height_hist(df):
    fig, ax = styled_fig()
    sns.histplot(df['Height'].dropna(), bins=30, kde=True, ax=ax, color='#FFBF00')
    ax.set_title("Height Distribution")
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    return fig

def plot_box_bmi_by_gender(df):
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(8, 4))

    # Base boxplot with no outliers shown
    sns.boxplot(
        x='Gender', y='BMI', data=df,
        ax=ax,
        palette='colorblind',
        showfliers=False
    )

    # Define outlier condition using IQR
    outlier_data = pd.DataFrame()
    for gender in df['Gender'].unique():
        gender_data = df[df['Gender'] == gender]
        q1 = gender_data['BMI'].quantile(0.25)
        q3 = gender_data['BMI'].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = gender_data[(gender_data['BMI'] < lower) | (gender_data['BMI'] > upper)]
        outlier_data = pd.concat([outlier_data, outliers])

    # Plot outliers as scatter points on top
    gender_positions = {'Male': 0, 'Female': 1}
    palette = sns.color_palette("colorblind")

    for gender in outlier_data['Gender'].unique():
        gender_outliers = outlier_data[outlier_data['Gender'] == gender]
        color = palette[0] if gender == 'Male' else palette[1]
        x_pos = gender_positions[gender]
        ax.scatter(
            np.full_like(gender_outliers['BMI'], x_pos, dtype=float),
            gender_outliers['BMI'],
            color=color,
            edgecolor='white',
            s=28,
            alpha=0.9,
            linewidth=0.4,
            zorder=3
        )

    ax.set_title('Boxplot of BMI by Gender')
    plt.tight_layout()
    return fig

def plot_bmi_by_agegroup(df):
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 18, 30, 45, 60, 75, 90],
                            labels=['0–18', '19–30', '31–45', '46–60', '61–75', '76+'], right=False)
    grouped = df.groupby('AgeGroup')['BMI'].mean().reset_index()
    fig, ax = styled_fig()
    sns.barplot(x='AgeGroup', y='BMI', data=grouped, ax=ax, palette='colorblind')
    ax.set_title('Average BMI by Age Group')
    plt.tight_layout()
    return fig

def plot_bmi_by_gender_bar(df):
    fig, ax = styled_fig()
    sns.barplot(x='Gender', y='BMI', data=df, ax=ax, palette='colorblind')
    ax.set_title('Average BMI by Gender')
    plt.tight_layout()
    return fig

def plot_bmi_category_pie(df):
    def categorize_bmi(bmi):
        if bmi < 18.5: return 'Underweight'
        elif bmi < 25: return 'Normal'
        elif bmi < 30: return 'Overweight'
        return 'Obese'

    df['BMICategory'] = df['BMI'].apply(categorize_bmi)
    counts = df['BMICategory'].value_counts()

    category_order = ['Underweight', 'Normal', 'Overweight', 'Obese']
    colorblind_colors = sns.color_palette("colorblind", 4)  # Use first 4 for pie

    fig, ax = plt.subplots(figsize=(4.2, 3.2))

    wedges, texts, autotexts = ax.pie(
        [counts.get(cat, 0) for cat in category_order],
        labels=category_order,
        colors=colorblind_colors,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'color': 'white', 'fontsize': 8},
        wedgeprops={'linewidth': 1, 'edgecolor': '#1E1E1E'}
    )

    ax.set_title('BMI Category Distribution', color='#FFBF00', fontsize=12)
    plt.tight_layout()
    return fig

    ax.set_title('BMI Category Distribution', color='#FFBF00', fontsize=12)
    plt.tight_layout()
    return fig

def plot_bmi_category_by_gender(df):
    df['BMICategory'] = df['BMI'].apply(lambda x: 'Underweight' if x < 18.5 else
                                                    'Normal' if x < 25 else
                                                    'Overweight' if x < 30 else
                                                    'Obese')
    fig, ax = styled_fig()
    sns.countplot(x='BMICategory', hue='Gender', data=df, ax=ax, palette='colorblind')
    ax.set_title('BMI Category by Gender')
    plt.tight_layout()
    return fig
#Trends
def plot_bmi_vs_age(df):
    fig, ax = styled_fig()
    sns.scatterplot(
        data=df, x='Age', y='BMI',
        ax=ax,
        color='#FBC02D',
        edgecolor='white',
        linewidth=0.3,
        s=10,
        alpha=0.5
    )
    ax.set_title('BMI vs Age')
    plt.tight_layout()
    return fig

def plot_bmi_age_trend(df):
    fig, ax = styled_fig()
    sns.regplot(
        data=df, x='Age', y='BMI',
        scatter_kws={'s': 10, 'alpha': 0.4, 'edgecolor': 'white', 'linewidths': 0.3},
        line_kws={'color': '#00E5FF', 'linewidth': 2},
        ax=ax,
        color='#FBC02D'
    )
    ax.set_title('BMI vs Age with Trend Line')
    plt.tight_layout()
    return fig


def plot_bmi_diastolic(df):
    fig, ax = styled_fig()
    sns.regplot(
        data=df, x='BMI', y='Diastolic BP',
        scatter_kws={'s': 10, 'alpha': 0.3, 'edgecolor': 'white', 'linewidths': 0.3},
        line_kws={'color': '#00E5FF', 'linewidth': 2},
        ax=ax,
        color='#FBC02D'
    )
    ax.set_title('BMI vs Diastolic BP')
    plt.tight_layout()
    return fig


def plot_bmi_systolic_trend(df):
    fig, ax = styled_fig()
    sns.scatterplot(
        data=df, x='BMI', y='Systolic BP',
        color='#FBC02D',
        edgecolor='white',
        s=10,
        alpha=0.4,
        ax=ax
    )
    z = np.polyfit(df['BMI'].dropna(), df['Systolic BP'].dropna(), 1)
    p = np.poly1d(z)
    ax.plot(df['BMI'], p(df['BMI']), color='#00E5FF', linewidth=2)
    ax.set_title('BMI vs Systolic Blood Pressure')
    plt.tight_layout()
    return fig

#Gender Graphs
def plot_kde_gender(df):
    fig, ax = plt.subplots(figsize=(7, 3.2))  # ⬆️ Slightly taller
    sns.kdeplot(
        data=df, x='BMI', hue='Gender',
        fill=True,
        alpha=0.4,
        palette='colorblind',
        ax=ax
    )
    ax.set_title('KDE of BMI by Gender')
    plt.tight_layout()
    return fig

def plot_violin_bmi_gender(df):
    fig, ax = plt.subplots(figsize=(7, 3.2))  # ⬆️ Matches KDE height
    sns.violinplot(
        x='Gender', y='BMI',
        data=df,
        ax=ax,
        palette='colorblind',
        inner='box'
    )
    ax.set_title('Violin Plot of BMI by Gender')
    plt.tight_layout()
    return fig

def plot_gender_bmi_strip(df):
    fig, ax = plt.subplots(figsize=(7, 3.4))  # back to natural, balanced
    sns.stripplot(
        x='Gender', y='BMI',
        data=df,
        palette='colorblind',
        size=2.8,                 # slightly bigger
        jitter=True,
        alpha=0.4,
        edgecolor='white',
        linewidth=0.25,
        ax=ax
    )
    ax.set_title("BMI Distribution by Gender (Stripplot)")
    plt.tight_layout()
    return fig


def plot_corr_heatmap(df):
    df_corr = df[['BMI', 'Weight', 'Height', 'Waist', 'Age', 'Systolic BP', 'Diastolic BP']].dropna()
    corr_matrix = df_corr.corr()

    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        cbar=True,
        square=True,
        annot_kws={"fontsize": 5, "color": "black"},
        ax=ax,
        cbar_kws = {'shrink': 0.7}
    )
    # Customize colorbar tick label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=5, colors='white')  # Adjust font size here

    ax.set_facecolor('#1E1E1E')
    fig.patch.set_facecolor('#1E1E1E')
    plt.xticks(rotation=0, fontsize=4, color='white')
    plt.yticks(rotation=0, fontsize=4, color='white')
    ax.tick_params(axis='both', colors='white')
    plt.tight_layout()
    return fig

def plot_pairplot(df):
    df_pair = df[['BMI', 'Age', 'Systolic BP', 'Diastolic BP', 'Weight', 'Waist']].dropna()

    pair = sns.pairplot(
        df_pair,
        corner=True,
        plot_kws={
            "s": 7,                        # ⬅️ Small dots
            "alpha": 0.5,                  # ⬅️ Slight transparency
            "edgecolor": "#1E1E1E",        # ⬅️ Border match background
            "linewidth": 0.3               # ⬅️ Fine edge
        },
        diag_kws={
            "bins": 30,
            "color": "#FFBF00",            # ⬅️ Amber histo bars
            "edgecolor": "#1E1E1E"
        }
    )

    pair.fig.set_size_inches(7, 6.5)
    pair.fig.set_dpi(120)  # ⬅️ BOOSTS clarity (important!)

    pair.fig.set_facecolor("#1E1E1E")
    for ax in pair.axes.flatten():
        if ax is not None:
            ax.set_facecolor("#1E1E1E")
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#FFBF00")

    return pair.fig

def plot_waist_age_trend(df):
    fig, ax = styled_fig()
    sns.regplot(
        data=df, x='Age', y='Waist',
        scatter_kws={
            's': 10,
            'alpha': 0.4,
            'edgecolor': 'white',
            'linewidths': 0.3  # ✅ use only one of 'linewidth' or 'linewidths'
        },
        line_kws={'color': '#00E5FF', 'linewidth': 2},
        ax=ax,
        color='#FBC02D'  # amber dots
    )
    ax.set_title('Waist Circumference vs Age')
    plt.tight_layout()
    return fig
def plot_waist_bmi(df):
    fig, ax = styled_fig()
    sns.scatterplot(
        data=df, x='Waist', y='BMI',
        color='#FBC02D',              # amber points
        edgecolor='white',            # slight edge for clarity
        linewidth=0.3,                # thin border
        s=10,                         # small dot size
        alpha=0.5,                    # moderate transparency
        ax=ax
    )

    z = np.polyfit(df['Waist'], df['BMI'], 1)
    p = np.poly1d(z)
    ax.plot(df['Waist'], p(df['Waist']), color='#00E5FF', linewidth=2)

    ax.set_title('Waist Circumference vs BMI')
    plt.tight_layout()
    return fig



# ---main runner
def run_all_eda_plots(df):
    st.markdown("""
    <style>
        .eda-section {
            background-color: #2A2A2A;
            border: 1px solid #FFBF00;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        h3 {
            color: #FFBF00;
            text-align: center;
            font-size: 22px;
        }
    </style>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "📊 Distribution", "📦 Categorical", "📈 Trends",
        "🎯 Gender", "🔍 Correlation", "🧍 Waist", "📌 Pairwise"
    ])

    with tabs[0]:
        st.markdown("<div class='eda-section'><h3>Distribution Plots</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_bmi_hist(df), use_container_width=True)
        with col2:
            st.pyplot(plot_weight_hist(df), use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            st.pyplot(plot_height_hist(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown("<div class='eda-section'><h3>Categorical Comparison</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_box_bmi_by_gender(df), use_container_width=True)
        with col2:
            st.pyplot(plot_bmi_by_agegroup(df), use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            st.pyplot(plot_bmi_by_gender_bar(df), use_container_width=True)
        with col4:
            st.pyplot(plot_bmi_category_by_gender(df), use_container_width=True)

        # ✅ Pie chart moved to full width below
        st.pyplot(plot_bmi_category_pie(df), use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown("<div class='eda-section'><h3>BMI Trends & Relationships</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_bmi_vs_age(df), use_container_width=True)
        with col2:
            st.pyplot(plot_bmi_age_trend(df), use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            st.pyplot(plot_bmi_diastolic(df), use_container_width=True)
        with col4:
            st.pyplot(plot_bmi_systolic_trend(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown("<div class='eda-section'><h3>Gender-Based Distributions</h3>", unsafe_allow_html=True)

        # 🔁 Stack KDE and Violin vertically instead of side-by-side
        st.pyplot(plot_kde_gender(df), use_container_width=True)
        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        st.pyplot(plot_violin_bmi_gender(df), use_container_width=True)

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        st.pyplot(plot_gender_bmi_strip(df), use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[4]:
        st.markdown("<div class='eda-section'><h3>Correlation Heatmap</h3>", unsafe_allow_html=True)
        st.pyplot(plot_corr_heatmap(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    with tabs[5]:  # 🧍 Waist
        st.markdown("<div class='eda-section'><h3>Waist Circumference Trends</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_waist_age_trend(df), use_container_width=True)
        with col2:
            st.pyplot(plot_waist_bmi(df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[6]:
        st.markdown("<div class='eda-section'><h3>Pairwise Relationships</h3>", unsafe_allow_html=True)
        st.pyplot(plot_pairplot(df), use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)
