import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Market Strategy Dashboard", layout="wide", page_icon="👁️")
st.title("Product Transition & Market Strategy Dashboard")
st.write("A unified Decision Support System for cannibalization forecasting and retention lead generation.")

# --- 2. DATA CACHING ---
@st.cache_data
def load_data():
    df = pd.read_csv('market_data.csv')
    
    # Force the date column back into a proper datetime object 
    if 'thnbln_dt' in df.columns:
        df['thnbln_dt'] = pd.to_datetime(df['thnbln_dt'])
        
    return df

df_panel = load_data()

# ==========================================
# SECTION 1: LOG-LOG ELASTICITY CALCULATOR
# ==========================================
st.divider()
st.header("📉 Cannibalization Elasticity Calculator")
st.write("Predict legacy product volume retention based on new technology adoption.")

# Train Model 
cutoff_date_calc = pd.to_datetime('2024-12-01')
df_log = df_panel[df_panel['thnbln_dt'] >= cutoff_date_calc].copy()

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

    # Calculate Dynamic Defaults based on actual dataset scale
    median_hna = int(df_log['total_hna'].median())
    default_hna = median_hna if median_hna > 0 else 50000 
    
    median_new_tech = int(df_log['qty_clareon'].median())
    default_new_tech = median_new_tech if median_new_tech > 0 else 50
    max_new_tech = max(500, int(df_log['qty_clareon'].max() * 1.5))

    # UI Inputs
    col1, col2 = st.columns(2)
    with col1:
        hospital_size = st.number_input("Hospital Monthly Spend (Market Capacity)", value=default_hna)
    with col2:
        new_tech_units = st.slider("Forecasted New Technology Units Ordered", 0, max_new_tech, default_new_tech)

    # Prediction Math
    ln_new_input = np.log1p(new_tech_units)
    ln_size_input = np.log1p(hospital_size)
    
    # Predict and convert back from log scale
    ln_pred = calc_model.predict([[ln_new_input, ln_size_input]])[0]
    predicted_legacy = np.expm1(ln_pred)
    predicted_legacy = max(0, predicted_legacy)

    # Display Results
    st.metric(label="Predicted Legacy Units Retained", value=round(predicted_legacy))
    st.info(f"**Insight:** The current Cannibalization Elasticity is {elasticity:.2f}. The market is highly inelastic, meaning we are retaining legacy volume effectively despite the new launch.")
    st.caption(f"*(Developer View) Raw Log Output: {ln_pred:.4f} | Raw Unit Output: {predicted_legacy:.4f}*")


# ==========================================
# SECTION 2: SALES STRATEGY (DEFENSIVE PLAY)
# ==========================================
st.divider()
st.header("🎯 Defensive Sales Strategy & Lead Generation")
st.write("These hospitals have NOT switched to the new technology yet, but our Random Forest model predicts a high probability of transition. Prioritize these accounts to prevent competitor takeover.")

# Propensity Model Logic
cutoff_date_ml = pd.to_datetime('2025-01-01')

switched_2025 = df_panel[df_panel['thnbln_dt'] >= cutoff_date_ml].groupby('cust_name')['qty_clareon'].sum()
target = (switched_2025 > 0).astype(int).reset_index()
target.columns = ['cust_name', 'adopted_new_tech']

hist_data = df_panel[df_panel['thnbln_dt'] < cutoff_date_ml].copy()
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
    # Train the Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_ml, y_ml)
    df_ml['Switch Probability'] = rf_clf.predict_proba(X_ml)[:, 1]
    
    # Filter for accounts that haven't switched yet, sorted by highest risk
    leads = df_ml[df_ml['adopted_new_tech'] == 0].sort_values(by='Switch Probability', ascending=False)
    
    # Display the Dataframe natively in Streamlit
    st.dataframe(
        leads[['cust_name', 'Switch Probability', 'total_legacy', 'total_spend']].head(15), 
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("Not enough variance in the data to run the Random Forest model (e.g., no one has adopted yet, or everyone has).")
