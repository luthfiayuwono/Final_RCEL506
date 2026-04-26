import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Market Strategy Dashboard", layout="wide", page_icon="👁️")
st.title("Product Transition & Market Strategy Dashboard")

# --- 2. DATA CACHING ---
@st.cache_data
def load_data():
    # Ensure your file is named 'market_data.csv' in your GitHub repo
    df = pd.read_csv('market_data.csv')
    return df

df_panel = load_data()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Module:", 
                        ["Cannibalization Calculator", 
                         "Sales Strategy (Lead Gen)"])

# ==========================================
# MODULE 1: LOG-LOG ELASTICITY CALCULATOR
# ==========================================
if page == "Cannibalization Calculator":
    st.header("📉 Cannibalization Elasticity Calculator")
    st.write("Predict legacy product volume retention based on new technology adoption.")
    
    # Train Model
    df_log = df_panel[df_panel['thnbln_dt'] >= '2024-12-01'].copy()
    df_log['ln_legacy'] = np.log1p(df_log['qty_acrysof_iq']) # Mapping legacy to AcrySof
    df_log['ln_new_tech'] = np.log1p(df_log['qty_clareon']) # Mapping new tech to Clareon
    df_log['ln_total_hna'] = np.log1p(df_log['total_hna'])
    
    X = df_log[['ln_new_tech', 'ln_total_hna']]
    y = df_log['ln_legacy']
    calc_model = LinearRegression().fit(X, y)
    elasticity = calc_model.coef_[0]

    # UI Inputs
    col1, col2 = st.columns(2)
    with col1:
        hospital_size = st.number_input("Hospital Monthly Spend (Market Capacity)", value=50000, step=5000)
    with col2:
        new_tech_units = st.slider("Forecasted New Technology Units Ordered", 0, 500, 50)

    # Prediction Math
    ln_new_input = np.log1p(new_tech_units)
    ln_size_input = np.log1p(hospital_size)
    ln_pred = calc_model.predict([[ln_new_input, ln_size_input]])[0]
    predicted_legacy = np.expm1(ln_pred)

    st.divider()
    st.metric(label="Predicted Legacy Units Retained", value=int(predicted_legacy))
    st.info(f"**Insight:** The current Cannibalization Elasticity is {elasticity:.2f}. The market is highly inelastic, meaning we are retaining legacy volume effectively despite the new launch.")

# ==========================================
# MODULE 2: SALES STRATEGY (OFFENSE VS DEFENSE)
# ==========================================
elif page == "Sales Strategy (Lead Gen)":
    st.header("🎯 Sales Strategy & Lead Generation")
    
    strategy = st.radio("Select Strategy Playbook:", ["🛡️ Defensive (Retention/Propensity)", "⚔️ Offensive (Market Share Conquest)"])
    
    if strategy == "🛡️ Defensive (Retention/Propensity)":
        st.subheader("High-Risk Loyalists (Random Forest Classification)")
        st.write("These hospitals have NOT switched to the new technology yet, but our ML model predicts a high probability of transition. Prioritize these for retention.")
        
        # Propensity Model Logic
        switched_2025 = df_panel[df_panel['thnbln_dt'] >= '2025-01-01'].groupby('cust_name')['qty_clareon'].sum()
        target = (switched_2025 > 0).astype(int).reset_index()
        target.columns = ['cust_name', 'adopted_new_tech']
        
        hist_data = df_panel[df_panel['thnbln_dt'] < '2025-01-01'].copy()
        features = hist_data.groupby('cust_name').agg({
            'qty_acrysof_iq': ['sum', 'mean'], 
            'total_hna': ['sum'], 
        }).reset_index()
        features.columns = ['cust_name', 'total_legacy', 'avg_legacy', 'total_spend']
        features = features.fillna(0)
        
        df_ml = features.merge(target, on='cust_name', how='inner')
        X_ml = df_ml.drop(['cust_name', 'adopted_new_tech'], axis=1)
        y_ml = df_ml['adopted_new_tech']
        
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_ml, y_ml)
        df_ml['Switch Probability'] = rf_clf.predict_proba(X_ml)[:, 1]
        
        leads = df_ml[df_ml['adopted_new_tech'] == 0].sort_values(by='Switch Probability', ascending=False)
        st.dataframe(leads[['cust_name', 'Switch Probability', 'total_legacy']].head(15), use_container_width=True)

    else:
        st.subheader("White Space Targets (Heuristic Gap Analysis)")
        st.write("These hospitals have high spending power but buy very few of our units. They represent competitor-held territory.")
        
        # Conquest Model Logic
        df_share = df_panel.groupby('cust_name').agg({'qty_acrysof_iq': 'sum', 'qty_clareon': 'sum', 'total_hna': 'sum'}).reset_index()
        df_share['total_brand_units'] = df_share['qty_acrysof_iq'] + df_share['qty_clareon']
        avg_eff = df_share['total_brand_units'].sum() / df_share['total_hna'].sum()
        df_share['Expected Units'] = df_share['total_hna'] * avg_eff
        df_share['Unit Gap (Competitor Space)'] = df_share['Expected Units'] - df_share['total_brand_units']
        
        conquest = df_share[df_share['Unit Gap (Competitor Space)'] > 0].sort_values(by='Unit Gap (Competitor Space)', ascending=False)
        st.dataframe(conquest[['cust_name', 'total_hna', 'total_brand_units', 'Unit Gap (Competitor Space)']].head(15), use_container_width=True)
