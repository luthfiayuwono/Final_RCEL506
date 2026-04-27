import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. PAGE CONFIG & BRANDING
# ==========================================
st.set_page_config(page_title="Clareon Strategic Dashboard", page_icon="🎯", layout="wide")

# The "So What?" Header
st.title("New Product Launch Analytics: Strategic Decision Dashboard")
st.markdown("""
**Product Key:** 🔵 **Clareon:** New product launch | 🔘 **AcrySof:** Existing legacy product
""")

# ==========================================
# 2. DATA PIPELINE (CACHED)
# ==========================================
@st.cache_data
def load_data():
    url = "https://docs.google.com/spreadsheets/d/1MyD0f9PxRyWzKYqIspSn7vQHY0whyDAN/export?format=xlsx"
    response = requests.get(url)
    df = pd.read_excel(BytesIO(response.content))
    
    LAUNCH_DATE = '2024-12-01'
    target_products = ['ARCRYSOF IQ', 'CLAREON AUTONOME']
    df_targets = df[df['group_name'].isin(target_products)].copy()
    df_targets['thnbln'] = pd.to_datetime(df_targets['thnbln']).dt.to_period('M')

    df_panel = df_targets.pivot_table(
        index=['cust_name', 'cust_city', 'thnbln'], 
        columns='group_name', values='qty', aggfunc='sum', fill_value=0
    ).reset_index()
    df_panel = df_panel.rename(columns={'ARCRYSOF IQ': 'qty_acrysof', 'CLAREON AUTONOME': 'qty_clareon'})
    df_panel['thnbln_dt'] = df_panel['thnbln'].dt.to_timestamp()
    
    return df_targets, df_panel, LAUNCH_DATE

df_targets, df_panel, LAUNCH_DATE = load_data()

# ==========================================
# 3. SIDEBAR: GLOBAL FILTERS
# ==========================================
st.sidebar.header("🗺️ Global Filters")
all_cities = ["All Cities"] + sorted(df_panel['cust_city'].unique().tolist())
selected_city = st.sidebar.selectbox("Filter by Market/City:", all_cities)

# Apply Global Filter
if selected_city != "All Cities":
    filtered_df = df_panel[df_panel['cust_city'] == selected_city]
    filtered_targets = df_targets[df_targets['cust_city'] == selected_city]
else:
    filtered_df = df_panel
    filtered_targets = df_targets

# ==========================================
# 4. KPI SCORECARDS (AT-A-GLANCE)
# ==========================================
total_clareon = filtered_df['qty_clareon'].sum()
avg_legacy = filtered_df[filtered_df['thnbln_dt'] < LAUNCH_DATE]['qty_acrysof'].mean()
net_expansion_est = "95.0%" # Based on -0.05 elasticity

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Total Clareon Units", int(total_clareon), help="Total units sold since launch across filtered markets.")
kpi2.metric("Market Expansion Efficiency", net_expansion_est, help="Calculated as (1 - Elasticity Coefficient). High % indicates new growth.")
kpi3.metric("Avg. Legacy Volume", f"{avg_legacy:.1f}", help="Baseline monthly volume of AcrySof pre-launch.")

st.markdown("---")

# ==========================================
# 5. TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📈 Market Performance", "🧮 Growth Simulator", "🎯 Lead Action Plan"])

with tab1:
    st.subheader("Market Performance: Clareon is successfully expanding the footprint.")
    col1, col2 = st.columns(2)
    
    with col1:
        # Graph 1 logic with filtered data...
        st.write("**Product Crossover Trends**")
        graph1 = filtered_targets.pivot_table(index='thnbln', columns='group_name', values='qty', aggfunc='sum').fillna(0)
        graph1.index = graph1.index.to_timestamp()
        fig1, ax1 = plt.subplots(figsize=(8,4))
        ax1.plot(graph1.index, graph1.iloc[:, 0], color='Grey', marker='o', label='AcrySof')
        ax1.plot(graph1.index, graph1.iloc[:, 1], color='#007bff', marker='s', label='Clareon')
        ax1.legend()
        st.pyplot(fig1)

    with col2:
        st.write("**Market Share Distribution**")
        df_bar = filtered_targets.groupby(['thnbln', 'group_name'])['qty'].sum().unstack(fill_value=0)
        df_bar_pct = df_bar.div(df_bar.sum(axis=1), axis=0) * 100
        fig2, ax2 = plt.subplots(figsize=(8,4))
        df_bar_pct.plot(kind='bar', stacked=True, color=['Grey', '#007bff'], ax=ax2)
        st.pyplot(fig2)

with tab2:
    st.header("🧮 'What-If' Cannibalization Simulator")
    
    # User Input with Tooltip
    sim_input = st.number_input("Projected Clareon Sales Units:", value=100, 
                                help="Input the number of Clareon units you expect to sell to see the predicted impact on legacy products.")
    
    # Math logic
    displacement = sim_input * 0.05
    net_gain = sim_input - displacement
    
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Net New Growth", f"+{net_gain:.1f} Units")
    with c2:
        st.metric("Legacy Displacement", f"-{displacement:.1f} Units", delta_color="inverse")

    with st.expander("🔍 View Technical Methodology"):
        st.write("""
        Our **Fixed-Effects OLS Regression** calculated a cannibalization coefficient of **-0.05**. 
        This means that for every 100 units of Clareon sold, we statistically observe only a 5-unit 
        reduction in AcrySof. This suggests that the launch is driving market expansion rather 
        than simple substitution.
        """)

with tab3:
    st.header("🎯 Lead Action Plan")
    st.write("Targeting stagnant hospitals with the highest predicted propensity to switch.")
    
    # Creating a dummy Lead List for the demo (Replace with your actual ML output)
    leads = pd.DataFrame({
        "Hospital Name": ["Mercy General", "Saint Jude's", "City Eye Clinic", "Regional Health"],
        "Switch Probability": [0.88, 0.75, 0.62, 0.45],
        "Legacy Tier": ["Gold", "Silver", "Gold", "Bronze"],
        "Last Contact": ["2 days ago", "1 week ago", "2 weeks ago", "Never"]
    })

    # INTERACTIVE TABLE
    st.dataframe(
        leads,
        column_config={
            "Switch Probability": st.column_config.ProgressColumn(
                "Switch Probability",
                help="Predicted probability of switching based on Random Forest model",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Legacy Tier": st.column_config.SelectboxColumn(
                "Priority Status",
                options=["Gold", "Silver", "Bronze"],
            )
        },
        hide_index=True,
        use_container_width=True
    )
    st.download_button("📥 Export Lead List to Excel", data=leads.to_csv(), file_name="Clareon_Leads.csv")
