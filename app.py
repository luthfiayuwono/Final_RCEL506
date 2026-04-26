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
    df = pd.read_csv('market_data.csv')
    
    # BUG FIX: Force the date column back into a proper datetime object 
    # so the '2024-12-01' filter actually works!
    if 'thnbln_dt' in df.columns:
        df['thnbln_dt'] = pd.to_datetime(df['thnbln_dt'])
        
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
    
    # Train Model (Using proper datetime comparison)
    cutoff_date = pd.to_datetime('2024-12-01')
    df_log = df_panel[df_panel['thnbln_dt'] >= cutoff_date].copy()
    
    # Check if data exists to prevent errors
    if df_log.empty:
        st.error("Error: No data found after Dec 2024. Check your dataset dates.")
    else:
        df_log['ln_legacy'] = np.log1p(df_log['qty_acrysof_iq']) 
        df_log['ln_new_tech'] = np.log1p(df_log['qty_clareon']) 
        df_log['ln_total_hna'] = np.log1p(df_log['total_hna'])
        
        X = df_log[['ln_new_tech', 'ln_total_hna']]
        y = df_log['ln_legacy']
        calc_model = LinearRegression().fit(X, y)
        elasticity = calc_model.coef_[0]

        # UI Inputs
        col1, col2 = st.columns(2)
        with col1:
            # Note: If your total_hna is in IDR (millions/billions), increase this value!
            hospital_size = st.number_input("Hospital Monthly Spend (Market Capacity)", value=50000, step=5000)
        with col2:
            new_tech_units = st.slider("Forecasted New Technology Units Ordered", 0, 500, 50)

        # Prediction Math
        ln_new_input = np.log1p(new_tech_units)
        ln_size_input = np.log1p(hospital_size)
        
        # Predict and convert back from log scale
        ln_pred = calc_model.predict([[ln_new_input, ln_size_input]])[0]
        predicted_legacy = np.expm1(ln_pred)
        
        # Ensure we don't predict negative units due to extreme math
        predicted_legacy = max(0, predicted_legacy)

        st.divider()
        st.metric(label="Predicted Legacy Units Retained", value=int(predicted_legacy))
        st.info(f"**Insight:** The current Cannibalization Elasticity is {elasticity:.2f}. The market is highly inelastic, meaning we are retaining legacy volume effectively despite the new launch.")

# ==========================================
# MODULE 2: SALES STRATEGY (DEFENSIVE ONLY)
# ==========================================
elif page == "Sales Strategy (Lead Gen)":
    st.header("🎯 Sales Strategy & Lead Generation")
    
    st.subheader("🛡️ Defensive Play: High-Risk Loyalists (Random Forest)")
    st.write("These hospitals have NOT switched to the new technology yet, but our ML model predicts a high probability of transition. Prioritize these for retention to prevent competitor takeover.")
    
    # Propensity Model Logic
    cutoff_date = pd.to_datetime('2025-01-01')
    
    switched_2025 = df_panel[df_panel['thnbln_dt'] >= cutoff_date].groupby('cust_name')['qty_clareon'].sum()
    target = (switched_2025 > 0).astype(int).reset_index()
    target.columns = ['cust_name', 'adopted_new_tech']
    
    hist_data = df_panel[df_panel['thnbln_dt'] < cutoff_date].copy()
    features = hist_data.groupby('cust_name').agg({
        'qty_acrysof_iq': ['sum', 'mean'], 
        'total_hna': ['sum'], 
    }).reset_index()
    
    # Flatten column names safely
    features.columns = ['cust_name', 'total_legacy', 'avg_legacy', 'total_spend']
    features = features.fillna(0)
    
    df_ml = features.merge(target, on='cust_name', how='inner')
    X_ml = df_ml.drop(['cust_name', 'adopted_new_tech'], axis=1)
    y_ml = df_ml['adopted_new_tech']
    
    if not X_ml.empty and len(y_ml.unique()) > 1:
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_ml, y_ml)
        df_ml['Switch Probability'] = rf_clf.predict_proba(X_ml)[:, 1]
        
        leads = df_ml[df_ml['adopted_new_tech'] == 0].sort_values(by='Switch Probability', ascending=False)
        st.dataframe(leads[['cust_name', 'Switch Probability', 'total_legacy']].head(15), use_container_width=True)
    else:
        st.warning("Not enough variance in the data to run the Random Forest model (e.g., no one has adopted yet, or everyone has).")
