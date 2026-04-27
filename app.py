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
    rf_model.fit(X_pred
