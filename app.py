import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Clareon Launch: Predictive Propensity Model",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Clareon Launch: Predictive Analytics & Lead Routing")
st.markdown("""
**Executive Dashboard:** Identifying high-probability hospital targets and quantifying product cannibalization.
""")

# ==========================================
# 2. DATA PIPELINE (CACHED FOR SPEED)
# ==========================================
@st.cache_data
def load_and_prep_data():
    url = "https://docs.google.com/spreadsheets/d/1MyD0f9PxRyWzKYqIspSn7vQHY0whyDAN/export?format=xlsx"
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content))
    
    LAUNCH_DATE = '2024-12-01'
    
    target_products = ['ARCRYSOF IQ', 'CLAREON AUTONOME']
    df_targets = df[df['group_name'].isin(target_products)].copy()
    df_targets = df_targets[df_targets['qty'] > 0].copy()
    df_targets['thnbln'] = pd.to_datetime(df_targets['thnbln']).dt.to_period('M')

    df_panel_qty = df_targets.pivot_table(
        index=['cust_name', 'thnbln'],
        columns='group_name',
        values='qty',
        aggfunc='sum',
        fill_value=0
    ).reset_index()

    df_panel_qty.columns.name = None
    df_panel_qty = df_panel_qty.rename(columns={'ARCRYSOF IQ': 'qty_acrysof_iq', 'CLAREON AUTONOME': 'qty_clareon'})

    hospital_attr = df_targets[['cust_name', 'cust_city']].dropna().drop_duplicates(subset=['cust_name'])
    monthly_hna = df_targets.groupby(['cust_name', 'thnbln'])['total_hna'].sum().reset_index()

    df_panel = df_panel_qty.merge(hospital_attr, on='cust_name', how='left')
    df_panel = df_panel.merge(monthly_hna, on=['cust_name', 'thnbln'], how='left')
    df_panel['thnbln_dt'] = df_panel['thnbln'].dt.to_timestamp()
    
    return df_targets, df_panel, hospital_attr, LAUNCH_DATE

with st.spinner('Pulling live data from Data Warehouse...'):
    df_targets, df_panel, hospital_attr, LAUNCH_DATE = load_and_prep_data()

# ==========================================
# 3. MACHINE LEARNING PIPELINE
# ==========================================
@st.cache_resource
def train_propensity_model(df_panel, hospital_attr):
    pre_launch = df_panel[df_panel['thnbln_dt'] < LAUNCH_DATE].copy()
    post_launch = df_panel[df_panel['thnbln_dt'] >= LAUNCH_DATE].copy()

    switched_2025 = post_launch.groupby('cust_name')['qty_clareon'].sum()
    target = (switched_2025 > 0).astype(int).reset_index(name='adopted_clareon')

    features = pre_launch.groupby('cust_name').agg({
        'qty_acrysof_iq': ['sum', 'mean'],
        'total_hna': ['sum']
    }).reset_index()
    features.columns = ['cust_name', 'total_legacy', 'avg_legacy', 'total_spend']
    features = features.fillna(0)

    df_ml = features.merge(target, on='cust_name', how='inner')

    X_pred = df_ml.drop(['cust_name', 'adopted_clareon'], axis=1)
    y_pred = df_ml['adopted_clareon']

    # Final Production Model
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight={0: 1, 1: 1.2}, random_state=42)
    rf_model.fit(X_pred, y_pred)
    
    df_ml['switch_probability'] = rf_model.predict_proba(X_pred)[:, 1]
    df_ml = df_ml.merge(hospital_attr, on='cust_name', how='left')
    
    return df_ml

df_ml = train_propensity_model(df_panel, hospital_attr)

# ==========================================
# 4. SIDEBAR & INTERACTIVITY
# ==========================================
st.sidebar.header("⚙️ Operational Controls")
st.sidebar.markdown("Use this slider to adjust lead routing filters live:")

# LIVE DEMO FEATURE: This variable updates the whole app dynamically
min_volume = st.sidebar.slider("Minimum Legacy Volume (Units):", min_value=1, max_value=50, value=10, step=1)
prob_threshold = st.sidebar.slider("High Priority Probability Threshold:", min_value=0.50, max_value=0.80, value=0.65, step=0.05)

st.sidebar.info(f"Currently filtering for accounts with >{min_volume} historic units.")

# ==========================================
# 5. DASHBOARD TABS
# ==========================================
tab1, tab2 = st.tabs(["📊 Market Trends (EDA)", "🎯 Prescriptive Lead List"])

with tab1:
    st.subheader("Graph 1: Product Crossover")
    
    # Graph 1 Logic
    graph1 = df_targets.pivot_table(index='thnbln', columns='group_name', values='qty', aggfunc='sum').fillna(0)
    graph1.columns = ['AcrySof (Legacy)', 'Clareon (New)']
    graph1.index = graph1.index.to_timestamp()

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(graph1.index, graph1['AcrySof (Legacy)'], color='Grey', linewidth=3, marker='o', label='AcrySof (Legacy)')
    ax1.plot(graph1.index, graph1['Clareon (New)'], color='Blue', linewidth=3, marker='s', label='Clareon (New Launch)')

    launch_date_dt = pd.to_datetime('2024-12-01')
    ax1.axvline(launch_date_dt, color='black', linewidth=1.5, linestyle='--')
    ax1.text(launch_date_dt, ax1.get_ylim()[1]*0.85, '  Clareon Launch', fontweight='bold', fontsize=10)

    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
    plt.xticks(rotation=45, ha='right')
    ax1.set_ylim(bottom=0)
    ax1.set_ylabel('Units Sold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig1)

with tab2:
    st.subheader("Prescriptive Analytics: Sales Routing Engine")
    st.markdown("This table prioritizes hospitals that have **not yet switched** based on their Random Forest propensity score.")
    
    # Prescriptive Logic using the Interactive Sidebar Controls
    lead_list = df_ml[
        (df_ml['adopted_clareon'] == 0) &
        (df_ml['total_legacy'] >= min_volume)
    ].sort_values(by=['switch_probability', 'total_legacy'], ascending=[False, False]).copy()

    def assign_action(prob):
        if prob >= prob_threshold: return "🚨 Immediate Priority Call"
        elif prob >= (prob_threshold - 0.10): return "⚠️ Secondary Target"
        else: return "Strategic Nurture"

    lead_list['Prescribed_Action'] = lead_list['switch_probability'].apply(assign_action)
    lead_list['switch_probability'] = (lead_list['switch_probability'] * 100).round(1).astype(str) + '%'

    final_hit_list = lead_list[[
        'cust_name', 'cust_city', 'switch_probability',
        'total_legacy', 'total_spend', 'Prescribed_Action'
    ]]
    
    st.dataframe(final_hit_list, use_container_width=True, hide_index=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Qualified Leads", len(final_hit_list))
    col2.metric("High Priority Targets", len(final_hit_list[final_hit_list['Prescribed_Action'].str.contains("Immediate")]))
    col3.metric("Model Baseline", "Random Forest")
