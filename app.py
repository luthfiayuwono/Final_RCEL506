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
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Clareon Analytics Suite", page_icon="👁️‍🗨️", layout="wide")
st.title("👁️‍🗨️ Clareon Launch Analytics: Strategic Decision Suite")
st.markdown("---")

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
    df_targets = df_targets[df_targets['qty'] > 0].copy()
    df_targets['thnbln'] = pd.to_datetime(df_targets['thnbln']).dt.to_period('M')

    df_panel_qty = df_targets.pivot_table(
        index=['cust_name', 'thnbln'], columns='group_name', values='qty', aggfunc='sum', fill_value=0
    ).reset_index()
    df_panel_qty.columns.name = None
    df_panel_qty = df_panel_qty.rename(columns={'ARCRYSOF IQ': 'qty_acrysof_iq', 'CLAREON AUTONOME': 'qty_clareon'})

    hospital_attr = df_targets[['cust_name', 'cust_city']].dropna().drop_duplicates(subset=['cust_name'])
    monthly_hna = df_targets.groupby(['cust_name', 'thnbln'])['total_hna'].sum().reset_index()

    df_panel = df_panel_qty.merge(hospital_attr, on='cust_name', how='left')
    df_panel = df_panel.merge(monthly_hna, on=['cust_name', 'thnbln'], how='left')
    df_panel['thnbln_dt'] = df_panel['thnbln'].dt.to_timestamp()
    
    return df_targets, df_panel, hospital_attr, LAUNCH_DATE

# ==========================================
# 3. MACHINE LEARNING PIPELINE (CACHED)
# ==========================================
@st.cache_resource
def train_propensity_model(df_panel, hospital_attr, LAUNCH_DATE):
    pre_launch = df_panel[df_panel['thnbln_dt'] < LAUNCH_DATE].copy()
    post_launch = df_panel[df_panel['thnbln_dt'] >= LAUNCH_DATE].copy()

    switched_2025 = post_launch.groupby('cust_name')['qty_clareon'].sum()
    target = (switched_2025 > 0).astype(int).reset_index(name='adopted_clareon')

    features = pre_launch.groupby('cust_name').agg({
        'qty_acrysof_iq': ['sum', 'mean'], 'total_hna': ['sum']
    }).reset_index()
    features.columns = ['cust_name', 'total_legacy', 'avg_legacy', 'total_spend']
    features = features.fillna(0)

    df_ml = features.merge(target, on='cust_name', how='inner')

    X_pred = df_ml.drop(['cust_name', 'adopted_clareon'], axis=1)
    y_pred = df_ml['adopted_clareon']

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight={0: 1, 1: 1.2}, random_state=42)
    rf_model.fit(X_pred, y_pred)
    
    df_ml['switch_probability'] = rf_model.predict_proba(X_pred)[:, 1]
    df_ml = df_ml.merge(hospital_attr, on='cust_name', how='left')
    return df_ml

# Load Data & Model
with st.spinner("Initializing Data & Machine Learning Engine..."):
    df_targets, df_panel, hospital_attr, LAUNCH_DATE = load_data()
    df_ml = train_propensity_model(df_panel, hospital_attr, LAUNCH_DATE)

# ==========================================
# 4. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📈 Market Visuals", "🧮 Cannibalization Calculator", "🎯 Lead Routing Engine"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Graph 1: Product Crossover")
        graph1 = df_targets.pivot_table(index='thnbln', columns='group_name', values='qty', aggfunc='sum').fillna(0)
        graph1.columns = ['AcrySof (Legacy)', 'Clareon (New)']
        graph1.index = graph1.index.to_timestamp()

        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(graph1.index, graph1['AcrySof (Legacy)'], color='Grey', linewidth=3, marker='o', label='AcrySof (Legacy)')
        ax1.plot(graph1.index, graph1['Clareon (New)'], color='Blue', linewidth=3, marker='s', label='Clareon (New Launch)')
        ax1.axvline(pd.to_datetime('2024-12-01'), color='black', linewidth=1.5, linestyle='--')
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        plt.xticks(rotation=45, ha='right')
        ax1.set_ylim(bottom=0)
        ax1.legend(loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig1)

    with col2:
        st.subheader("Graph 2: Market Share Shift")
        df_bar = df_targets.groupby(['thnbln', 'group_name'])['qty'].sum().unstack(fill_value=0)
        df_bar_pct = df_bar.div(df_bar.sum(axis=1), axis=0) * 100
        df_bar_pct = df_bar_pct[['ARCRYSOF IQ', 'CLAREON AUTONOME']]

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        df_bar_pct.plot(kind='bar', stacked=True, color=['Grey', 'Blue'], ax=ax2, width=0.85)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.xticks(rotation=45, ha='right')
        ax2.legend(['AcrySof', 'Clareon'], loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2)
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("Graph 3: Δ (Delta) Correlation Scatter Plot")
    
    # Exact Graph 3 from your code
    df_scatter = df_panel.sort_values(['cust_name', 'thnbln'])
    df_scatter['delta_acrysof'] = df_scatter.groupby('cust_name')['qty_acrysof_iq'].diff()
    df_scatter['delta_clareon'] = df_scatter.groupby('cust_name')['qty_clareon'].diff()

    df_plot = df_scatter.dropna().copy()
    df_plot = df_plot[(df_plot['delta_clareon'] > 0) | (df_plot['delta_clareon'] < 0)]

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.scatter(df_plot['delta_clareon'], df_plot['delta_acrysof'],
                alpha=0.6, color='blue', edgecolor='white', s=80, label='Hospitals')

    ax3.set_xlabel('Change in Clareon Units (Δ New Launch)', fontsize=10)
    ax3.set_ylabel('Change in AcrySof Units (Δ Legacy)', fontsize=10)
    ax3.axhline(0, color='black', linewidth=1.5, linestyle='--')
    ax3.axvline(0, color='black', linewidth=1.5, linestyle='--')
    ax3.axvspan(0, df_plot['delta_clareon'].max() * 1.05, ymin=0, ymax=0.5, color='red', alpha=0.05)
    
    ax3.text(df_plot['delta_clareon'].max() * 0.5, df_plot['delta_acrysof'].min() * 0.5,
             'Cannibalization\nQuadrant', fontsize=12, color='red', weight='bold', ha='center', alpha=0.5)
    ax3.legend(loc='upper right', frameon=True)
    ax3.grid(True, linestyle=':', alpha=0.6)
    st.pyplot(fig3)

with tab2:
    st.header("🧮 Decision Support: Cannibalization Calculator")
    st.info("Based on our Fixed-Effects Regression, the elasticity coefficient is **-0.05**.")
    
    # INTERACTIVE INPUT HERE
    sim_volume = st.number_input("Enter Projected Clareon Sales (Units):", min_value=0, max_value=10000, value=100, step=10)
    
    displacement = sim_volume * 0.05
    net_gain = sim_volume - displacement
    
    c1, c2, c3 = st.columns(3)
    c1.metric("New Clareon Volume", f"+{sim_volume} units")
    c2.metric("AcrySof Displacement", f"-{displacement:.1f} units", delta_color="inverse")
    c3.metric("Net Market Expansion", f"+{net_gain:.1f} units")
    
    efficiency_rate = (net_gain/sim_volume)*100 if sim_volume > 0 else 0
    st.markdown(f"""
    ### Managerial Verdict:
    For every **{sim_volume}** units of Clareon we sell, we only expect to lose **{displacement:.1f}** units of legacy volume. 
    This results in a **{efficiency_rate:.1f}% Efficiency Rate**, proving this is a Market Expansion strategy, not a substitution strategy.
    """)

with tab3:
    st.header("🎯 Prescriptive Analytics: Lead Routing")
    st.write("Prioritizing stagnant hospitals based on Random Forest Machine Learning Propensity Scores.")
    
    # INTERACTIVE LEAD LIST FILTERS
    col_a, col_b = st.columns(2)
    with col_a:
        top_n = st.selectbox("Show Top 'N' Leads:", options=[3, 5, 10, 20, 50], index=2) # Default is 10
    with col_b:
        min_vol = st.slider("Minimum Historic Legacy Volume:", min_value=1, max_value=50, value=10)
    
    # Filter the Dataframe
    lead_list = df_ml[
        (df_ml['adopted_clareon'] == 0) &
        (df_ml['total_legacy'] >= min_vol)
    ].sort_values(by=['switch_probability', 'total_legacy'], ascending=[False, False]).copy()

    def assign_action(prob):
        if prob >= 0.65: return "🚨 Priority Call"
        elif prob >= 0.55: return "⚠️ Secondary Target"
        else: return "Nurture"

    lead_list['Prescribed_Action'] = lead_list['switch_probability'].apply(assign_action)
    lead_list['switch_probability'] = (lead_list['switch_probability'] * 100).round(1).astype(str) + '%'

    final_hit_list = lead_list[[
        'cust_name', 'cust_city', 'switch_probability',
        'total_legacy', 'total_spend', 'Prescribed_Action'
    ]].head(top_n) # Apply the Top N filter here
    
    # Display the actual dataframe
    st.dataframe(final_hit_list, use_container_width=True, hide_index=True)
