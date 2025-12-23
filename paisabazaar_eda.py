import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PaisaBazaar Fraud Analysis",
    page_icon="🕵️",
    layout="wide"
)

# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    h1 {
        color: #2E86C1;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADING (Cached for performance) ---
@st.cache_data
def load_data():
    # Using the public export link method from your notebook
    sheet_id = "16YPPS76sdTNghDJ01zbsKmIZSpSwxD8Nd1FLGBdrwnM"
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    
    try:
        df = pd.read_csv(url)
        
        # Cleaning: Standardize column names
        df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()
        
        # Replace missing value placeholders
        for i in ['missing', 'null', 'miss', 'Not available']:
            df.replace({i: np.nan}, inplace=True)
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

df_raw = load_data()

# --- 2. DATA WRANGLING ---
@st.cache_data
def process_data(df):
    if df.empty:
        return pd.DataFrame()

    def most_common(rating_c):
        return rating_c.value_counts().idxmax()

    # Numeric Aggregation
    df_numerical = df.groupby('customer_id')[['age','annual_income','num_bank_accounts',
           'num_credit_card', 'interest_rate', 'num_of_loan',
           'delay_from_due_date', 'num_of_delayed_payment', 'changed_credit_limit',
           'num_credit_inquiries','outstanding_debt',
           'credit_utilization_ratio', 'credit_history_age','total_emi_per_month',
           'amount_invested_monthly','monthly_balance']].median().reset_index()

    # Categorical Aggregation
    df_categorical = df.groupby('customer_id')[['name','occupation', 'payment_of_min_amount', 
                                              'credit_mix', 'credit_score']].agg(most_common).reset_index()

    # Merge
    df_processed = pd.merge(df_numerical, df_categorical, on='customer_id')

    # Feature Engineering
    df_processed['debt_burden_ratio'] = df_processed['total_emi_per_month'] / (df_processed['annual_income'] / 12)
    
    return df_processed

df_processed = process_data(df_raw)

# --- SIDEBAR FILTERS ---
st.sidebar.header("⚙️ Data Filters")
st.sidebar.write("Interact with the dashboard by applying filters below.")

if not df_processed.empty:
    score_filter = st.sidebar.multiselect(
        "Select Credit Score:",
        options=df_processed['credit_score'].unique(),
        default=df_processed['credit_score'].unique()
    )
    
    # Filter the dataframe based on selection
    df_filtered = df_processed[df_processed['credit_score'].isin(score_filter)]
else:
    st.stop()

# --- MAIN LAYOUT ---
st.title("🕵️ Fraud Analysis for PaisaBazaar")
st.markdown("**EDA of Credit Scores based on customer data for detecting potential risks.**")

# Create Tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📝 Summary", "📊 Univariate", "📈 Bivariate", "🔗 Correlations", "🎬 Conclusion"])

# --- TAB 1: SUMMARY & DATA ---
with tab1:
    st.header("Project Summary")
    st.write("""
    This dashboard analyzes customer data to enhance credit assessment processes. 
    It explores variables like annual income, outstanding debt, and payment behavior to identify 
    patterns related to credit scores and potential fraud.
    """)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers (Unique)", df_processed['customer_id'].nunique())
    col2.metric("Avg Annual Income", f"${df_filtered['annual_income'].mean():,.2f}")
    col3.metric("Avg Outstanding Debt", f"${df_filtered['outstanding_debt'].mean():,.2f}")

    st.subheader("Preview Processed Data")
    st.dataframe(df_filtered.head())
    
    st.subheader("Numerical Data Distribution")
    st.write(df_filtered.describe())

    st.subheader("Categorical Data Distribution")
    st.write(df_filtered.describe(include='object'))

# --- TAB 2: UNIVARIATE ANALYSIS ---
with tab2:
    st.header("Univariate Analysis")
    
    col_type = st.radio("Select Column Type:", ["Numerical", "Categorical"], horizontal=True)
    
    if col_type == "Numerical":
        num_cols = ['num_credit_card', 'interest_rate', 'num_of_loan',
       'delay_from_due_date', 'num_of_delayed_payment', 'changed_credit_limit',
       'num_credit_inquiries', 'outstanding_debt', 'credit_utilization_ratio',
       'credit_history_age','debt_burden_ratio']
        
        selected_col = st.selectbox("Select Feature to Visualize:", num_cols)
        
        # Interactive Plotly Histogram
        fig = px.histogram(df_filtered, x=selected_col,
                           title=f"Distribution of {selected_col}", marginal="box", barmode="overlay")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        cat_cols = ['occupation', 'payment_of_min_amount', 'credit_mix', 'credit_score']
        selected_col = st.selectbox("Select Feature to Visualize:", cat_cols)
        
        # Interactive Plotly Bar Chart
        fig = px.histogram(df_filtered, x=selected_col, color=selected_col, 
                           title=f"Count of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: BIVARIATE ANALYSIS ---
with tab3:
    st.header("Bivariate Analysis")
    st.markdown("Explore relationships between Credit Score and other features.")
    
    # 1. Credit Mix & Min Payment
    st.subheader("Categorical Relationships")
    biv_opt = st.selectbox("Compare Credit Score with:", ['credit_mix', 'payment_of_min_amount'])
    
    fig_cat = px.histogram(df_filtered, x='credit_score', color=biv_opt, barmode='group',
                           title=f"Credit Score distribution by {biv_opt}")
    st.plotly_chart(fig_cat, use_container_width=True)
    
    st.divider()
    
    # 2. Numerical Relationships
    st.subheader("Numerical Relationships")
    y_axis_val = st.selectbox("Select Numerical Metric (Y-axis):", 
                              ['outstanding_debt', 'num_of_delayed_payment', 'interest_rate', 
                               'debt_burden_ratio', 'delay_from_due_date',
                               'num_credit_inquiries', 'credit_history_age',
                               'num_of_loan', 'changed_credit_limit'])
    
    # Box Plot
    fig_box = px.box(df_filtered, x='credit_score', y=y_axis_val, color='credit_score',
                        title=f"{y_axis_val} vs Credit Score")
    st.plotly_chart(fig_box, use_container_width=True)
        

# --- TAB 4: CORRELATIONS ---
with tab4:
    st.header("Multivariate Analysis")
    
    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    corr_cols = ['interest_rate','outstanding_debt', 'num_of_delayed_payment', 
                 'changed_credit_limit', 'num_credit_card', 'num_credit_inquiries',
                 'annual_income','credit_history_age', 'age', 'debt_burden_ratio']
    
    if len(df_filtered) > 0:
        corr_matrix = df_filtered[corr_cols].corr()
        fig_heat = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Not enough data to generate heatmap.")


    st.subheader("Scatter Plot Matrix")
    st.write("Visualizing interactions between key features.")

    # A. Create a temporary subset with shorter names for cleaner plotting
    plot_df = df_filtered.sample(min(500, len(df_filtered))).copy()
    
    # Mapping to shorter names for the plot axes
    rename_map = {
        'num_of_loan': 'Loans',
        'outstanding_debt': 'Debt',
        'num_credit_inquiries': 'Inquiries',
        'delay_from_due_date': 'Delay Days',
        'credit_score': 'Score'
    }
    plot_df.rename(columns=rename_map, inplace=True)
    
    dimensions = ['Loans', 'Debt', 'Inquiries', 'Delay Days']

    # B. Create the plot with specific height and marker adjustments
    fig_scp = px.scatter_matrix(
        plot_df,
        dimensions=dimensions,
        color='Score',
        color_discrete_sequence=["red", "green", "blue"],
        symbol='Score',
        title="Pairwise Relationships (Sampled Data)",
        height=900,  # <--- Crucial: Increases vertical space
        width=900    # <--- Optional: Helps if container width fails
    )

    # C. Tweak layout to prevent label overlapping
    fig_scp.update_traces(diagonal_visible=True, marker=dict(size=4, opacity=0.7))
    fig_scp.update_layout(
        font=dict(size=10), # Smaller font for axis labels
        margin=dict(l=40, r=40, t=60, b=40) # Adjust margins
    )
    
    st.plotly_chart(fig_scp, use_container_width=True)


# --- TAB 5: CONCLUSION ---
with tab5:
    st.header("Key Insights & Conclusion")
    
    st.success("### ✅ What Drives Credit Risk?")
    st.markdown("""
    Based on the analysis, the strongest indicators of poor credit scores are:
    * **Behavioral:** Frequent payment delays and poor credit-type choices.
    * **Financial:** High outstanding debt and reliance on minimum payments.
    * **History:** Shorter credit history age.
    """)
    
    st.warning("### ⚠️ Risk Signals")
    st.markdown("""
    * **Credit Mix:** Customers with 'Bad' credit mix almost universally have poor scores.
    * **Inquiries:** High number of credit inquiries correlates with risk.
    """)
    
    st.info("### 🚀 Recommendation")
    st.write("Future fraud detection models should prioritize behavioral features (delays, payment habits) over static demographics like Age or Occupation.")