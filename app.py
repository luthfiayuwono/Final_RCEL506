# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Alcon Market Strategy", layout="wide", page_icon="👁️")
st.title("Alcon: Product Transition & Market Strategy Dashboard")

# --- 2. DATA CACHING (So models don't retrain every click) ---
@st.cache_data
def load_data():
    # Load the data you saved from Colab
    df = pd.read_csv('alcon_data.csv')
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
    st.write("Predict legacy AcrySof volume retention based on new Clareon adoption.")
    
    # Train Model Quietly
    df_log = df_panel[df_panel['thnbln_dt'] >= '2024-12-01'].copy()
    df_log['ln_acrysof'] = np.log1p(df_log['qty_acrysof_iq'])
    df_log['ln_clareon'] = np.log1p(df_log['qty_clareon'])
    df_log['ln_total_hna'] = np.log1p(df_log['total_hna'])
    
    # Simple model for the calculator (Ignoring city for user simplicity)
    X = df_log[['ln_clareon', 'ln_total_hna']]
    y = df_log['ln_acrysof']
    calc_model = LinearRegression().fit(X, y)
    elasticity = calc_model.coef_[0]

    # UI Inputs
    col1, col2 = st.columns(2)
    with col1:
        hospital_size = st.number_input("Hospital Monthly Spend (HNA)", value=50000, step=5000)
    with col2:
        clareon_units = st.slider("Forecasted Clareon Units Ordered", 0, 500, 50)

    # Prediction Math
    ln_clareon_input = np.log1p(clareon_units)
    ln_size_input = np.log1p(hospital_size)
    ln_pred = calc_model.predict([[ln_clareon_input, ln_size_input]])[0]
    predicted_acrysof = np.expm1(ln_pred)

    st.divider()
    st.metric(label="Predicted AcrySof Retained Units", value=int(predicted_acrysof))
    st.info(f"**Insight:** The current Cannibalization Elasticity is {elasticity:.2f}. The market is highly inelastic, meaning we are retaining legacy volume better than a 1:1 swap.")

# ==========================================
# MODULE 2: SALES STRATEGY (OFFENSE VS DEFENSE)
# ==========================================
elif page == "Sales Strategy (Lead Gen)":
    st.header("🎯 Sales Strategy & Lead Generation")
    
    strategy = st.radio("Select Strategy Playbook:", ["🛡️ Defensive (Retention/Propensity)", "⚔️ Offensive (Market Share Conquest)"])
    
    if strategy == "🛡️ Defensive (Retention/Propensity)":
        st.subheader("High-Risk Loyalists (Random Forest Classification)")
        st.write("These hospitals have NOT switched to Clareon yet, but our ML model predicts they have the highest probability of doing so. Visit them before a competitor does.")
        
        # Propensity Model Logic
        switched_2025 = df_panel[df_panel['thnbln_dt'] >= '2025-01-01'].groupby('cust_name')['qty_clareon'].sum()
        target = (switched_2025 > 0).astype(int).reset_index()
        target.columns = ['cust_name', 'adopted_clareon']
        
        hist_data = df_panel[df_panel['thnbln_dt'] < '2025-01-01'].copy()
        features = hist_data.groupby('cust_name').agg({
            'qty_acrysof_iq': ['sum', 'mean'], 
            'total_hna': ['sum'], 
        }).reset_index()
        features.columns = ['cust_name', 'total_acrysof', 'avg_acrysof', 'total_spend']
        features = features.fillna(0)
        
        df_ml = features.merge(target, on='cust_name', how='inner')
        X_ml = df_ml.drop(['cust_name', 'adopted_clareon'], axis=1)
        y_ml = df_ml['adopted_clareon']
        
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_ml, y_ml)
        df_ml['Switch Probability'] = rf_clf.predict_proba(X_ml)[:, 1]
        
        leads = df_ml[df_ml['adopted_clareon'] == 0].sort_values(by='Switch Probability', ascending=False)
        st.dataframe(leads[['cust_name', 'Switch Probability', 'total_acrysof']].head(15), use_container_width=True)

    else:
        st.subheader("White Space Targets (Heuristic Gap Analysis)")
        st.write("These hospitals have massive spending power but buy very few Alcon units. They are buying from competitors. Use Clareon as a wedge to acquire them.")
        
        # Conquest Model Logic
        df_share = df_panel.groupby('cust_name').agg({'qty_acrysof_iq': 'sum', 'qty_clareon': 'sum', 'total_hna': 'sum'}).reset_index()
        df_share['total_alcon'] = df_share['qty_acrysof_iq'] + df_share['qty_clareon']
        avg_eff = df_share['total_alcon'].sum() / df_share['total_hna'].sum()
        df_share['Expected Units'] = df_share['total_hna'] * avg_eff
        df_share['Unit Gap (Competitor Space)'] = df_share['Expected Units'] - df_share['total_alcon']
        
        conquest = df_share[df_share['Unit Gap (Competitor Space)'] > 0].sort_values(by='Unit Gap (Competitor Space)', ascending=False)
        st.dataframe(conquest[['cust_name', 'total_hna', 'total_alcon', 'Unit Gap (Competitor Space)']].head(15), use_container_width=True)
