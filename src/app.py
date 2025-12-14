import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

# ---------------------------------------------------------
# 1. Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="A/B Test Decision Engine",
    page_icon="🧪",
    layout="wide"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .stMetric {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Helper Functions (Cached for Performance)
# ---------------------------------------------------------

@st.cache_data
def load_data(file):
    """Loads CSV data with caching to prevent reloading on every interaction."""
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data
def run_bootstrapping(df, group_col, metric_col, group_a, group_b, iterations=1000):
    """
    Performs bootstrapping to estimate the distribution of the difference in means.
    Returns the difference array.
    """
    boot_diffs = []
    for i in range(iterations):
        # Resample with replacement
        boot_sample = df.sample(frac=1, replace=True)
        # Calculate means for this sample
        boot_means = boot_sample.groupby(group_col)[metric_col].mean()
        # Calculate difference (A - B)
        diff = boot_means[group_a] - boot_means[group_b]
        boot_diffs.append(diff)
    
    return pd.DataFrame(boot_diffs, columns=['difference'])

# ---------------------------------------------------------
# 3. Sidebar Configuration
# ---------------------------------------------------------
st.sidebar.title("Configuration")
st.sidebar.markdown("Upload your experiment data to begin.")

uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=['csv'])

# Default dataset structure assumption (Cookie Cats)
default_group = 'version'
default_metrics = ['retention_1', 'retention_7']
default_continuous = 'sum_gamerounds'

# ---------------------------------------------------------
# 4. Main Application Logic
# ---------------------------------------------------------

st.title("Statistical A/B Test Decision Engine")
st.markdown("""
This tool automates statistical rigor for A/B testing. It moves beyond simple averages by using 
**Bootstrapping** for confidence intervals and **Mann-Whitney U** tests for skewed distributions.
""")

if uploaded_file is not None:
    # --- LOAD DATA ---
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        st.markdown("---")
        
        # --- DATA PREVIEW & MAPPING ---
        with st.expander("Data Preview & Settings", expanded=True):
            st.write("First 5 rows of uploaded data:")
            st.dataframe(df_raw.head())
            
            col1, col2, col3 = st.columns(3)
            with col1:
                # Try to auto-select if standard names exist
                g_idx = list(df_raw.columns).index(default_group) if default_group in df_raw.columns else 0
                group_col = st.selectbox("Group Column (Variant)", df_raw.columns, index=g_idx)
            
            with col2:
                # Filter for numeric/boolean columns for metrics
                numeric_cols = df_raw.select_dtypes(include=[np.number, bool]).columns.tolist()
                m_idx = numeric_cols.index(default_metrics[1]) if default_metrics[1] in numeric_cols else 0
                metric_col = st.selectbox("Primary Metric (e.g., Retention)", numeric_cols, index=m_idx)
                
            with col3:
                c_idx = numeric_cols.index(default_continuous) if default_continuous in numeric_cols else 0
                continuous_col = st.selectbox("Continuous Metric (e.g., Game Rounds)", numeric_cols, index=c_idx)

        # Get unique groups
        unique_groups = df_raw[group_col].unique()
        if len(unique_groups) != 2:
            st.error(f"Error: The Group Column must contain exactly 2 unique values. Found: {unique_groups}")
            st.stop()
            
        group_a = unique_groups[0] # e.g., gate_30
        group_b = unique_groups[1] # e.g., gate_40

        # --- DATA CLEANING (Outlier Removal) ---
        st.markdown("### Data Cleaning & Distributions")
        
        # Interactive Outlier Filter
        q99 = df_raw[continuous_col].quantile(0.99)
        threshold = st.slider(
            f"Filter {continuous_col} Outliers (Default < {int(q99)})", 
            min_value=int(df_raw[continuous_col].min()), 
            max_value=int(df_raw[continuous_col].max()), 
            value=3000 if 3000 < df_raw[continuous_col].max() else int(q99)
        )
        
        df_clean = df_raw[df_raw[continuous_col] < threshold].copy()
        removed_count = df_raw.shape[0] - df_clean.shape[0]
        
        st.caption(f"Removed {removed_count} rows (extreme outliers). Analyzing {df_clean.shape[0]:,} active players.")

        # Metric Overview
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Players", f"{df_clean.shape[0]:,}")
        col2.metric(f"Group A ({group_a})", f"{df_clean[df_clean[group_col]==group_a].shape[0]:,}")
        col3.metric(f"Group B ({group_b})", f"{df_clean[df_clean[group_col]==group_b].shape[0]:,}")

        # --- RETENTION ANALYSIS (BOOTSTRAPPING) ---
        st.markdown(f"### Retention Analysis: {metric_col}")
        st.info("Using **Bootstrapping (1000 iterations)** to visualize the certainty of the difference.")

        # Calculate Raw Rates
        rates = df_clean.groupby(group_col)[metric_col].mean()
        raw_diff = rates[group_a] - rates[group_b]
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric(f"{group_a} Rate", f"{rates[group_a]:.2%}")
        with col_b:
            st.metric(f"{group_b} Rate", f"{rates[group_b]:.2%}", delta=f"{-raw_diff:.2%}" if raw_diff > 0 else f"+{abs(raw_diff):.2%}", delta_color="inverse")

        # Run Bootstrapping
        with st.spinner("Running simulation..."):
            boot_df = run_bootstrapping(df_clean, group_col, metric_col, group_a, group_b)
        
        # Plotting Kernel Density Estimate (KDE)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.kdeplot(boot_df['difference'], fill=True, color='#FF4B4B', ax=ax)
        ax.axvline(0, color='black', linestyle='--', label="No Difference")
        ax.set_title(f"Bootstrap Distribution of Difference ({group_a} - {group_b})")
        ax.set_xlabel("Difference in Retention Rate")
        ax.legend()
        st.pyplot(fig)

        # Conclusion Logic
        prob_a_better = (boot_df['difference'] > 0).mean()
        st.write(f"**Probability that {group_a} is better than {group_b}:** `{prob_a_better:.1%}`")
        
        if prob_a_better > 0.95:
            st.success(f"Recommendation: **{group_a}** is the statistically significant winner!")
        elif prob_a_better < 0.05:
            st.success(f"Recommendation: **{group_b}** is the statistically significant winner!")
        else:
            st.warning("Recommendation: No significant difference detected. Stick to the control or investigate further.")

        # --- ENGAGEMENT ANALYSIS (MANN-WHITNEY U) ---
        st.markdown(f"### Engagement Analysis: {continuous_col}")
        st.markdown("Since game rounds are often skewed (power law distribution), we use the **Mann-Whitney U Test** instead of a T-Test.")

        # Mann-Whitney U Test
        u_stat, u_pval = stats.mannwhitneyu(
            df_clean[df_clean[group_col] == group_a][continuous_col],
            df_clean[df_clean[group_col] == group_b][continuous_col]
        )

        st.metric("Mann-Whitney p-value", f"{u_pval:.5f}")
        
        if u_pval < 0.05:
            st.write("**Result:** The distributions are significantly different.")
        else:
            st.write("**Result:** No significant difference in play habits.")

        # Visualization (Log Scale)
        st.write("Distribution of Game Rounds (Log Scale)")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.histplot(data=df_clean, x=continuous_col, hue=group_col, 
                     element="step", stat="density", common_norm=False, log_scale=True, ax=ax2)
        st.pyplot(fig2)

else:
    # Empty State
    st.info("👈 Please upload a CSV file from the sidebar to start the analysis.")
    st.markdown("""
    **Demo Data Format Expected:**
    - `userid`: Unique identifier
    - `version`: Group label (e.g., gate_30, gate_40)
    - `sum_gamerounds`: Continuous metric
    - `retention_1`: Boolean (0/1) or True/False
    - `retention_7`: Boolean (0/1) or True/False
    """)