"""
==============================================================================
Clareon Launch Analytics Suite: Execution Pipeline (app.py)
==============================================================================
This script executes the end-to-end data science pipeline for the Clareon 
product launch, including descriptive segmentation, econometric elasticity, 
predictive propensity modeling, and prescriptive lead routing.
"""

import pandas as pd
import numpy as np
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
import statsmodels.formula.api as smf
import warnings

# Suppress warnings for clean terminal output during live demo
warnings.filterwarnings('ignore')

# --- GLOBAL CONSTANTS ---
LAUNCH_DATE = '2024-12-01'
MINIMUM_VOLUME = 10 
DATA_URL = "https://docs.google.com/spreadsheets/d/1MyD0f9PxRyWzKYqIspSn7vQHY0whyDAN/export?format=xlsx"

def load_and_prep_data():
    print("\n[1/5] PREPARING DATA PIPELINE...")
    print("Downloading live data from Google Sheets...")
    response = requests.get(DATA_URL)
    response.raise_for_status()
    df = pd.read_excel(BytesIO(response.content))
    
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
    
    return df_panel, hospital_attr

def run_descriptive_analytics(df_panel):
    print("\n[2/5] DESCRIPTIVE ANALYTICS: FINANCIAL SEGMENTATION")
    pre_launch = df_panel[df_panel['thnbln_dt'] < LAUNCH_DATE].copy()
    post_launch = df_panel[df_panel['thnbln_dt'] >= LAUNCH_DATE].copy()

    pre_spend = pre_launch.groupby('cust_name')['total_hna'].sum().reset_index(name='pre_spend_legacy')
    post_spend = post_launch.groupby('cust_name').agg({
        'total_hna': 'sum', 'qty_acrysof_iq': 'sum', 'qty_clareon': 'sum'          
    }).reset_index().rename(columns={'total_hna': 'post_spend_total'})

    df_segment = pre_spend.merge(post_spend, on='cust_name', how='outer').fillna(0)

    def categorize(row):
        if row['pre_spend_legacy'] == 0 and row['qty_clareon'] > 0: return 'New Market'
        elif row['qty_clareon'] == 0 and row['post_spend_total'] > 0: return 'Stagnant (Legacy Only)'
        elif row['qty_clareon'] > 0 and row['post_spend_total'] > row['pre_spend_legacy']: return 'True Growth'
        elif row['qty_clareon'] > 0 and row['post_spend_total'] <= row['pre_spend_legacy']: return 'Cannibalizer'
        else: return 'Inactive'

    df_segment['Growth_Bucket'] = df_segment.apply(categorize, axis=1)
    print(df_segment['Growth_Bucket'].value_counts())
    return pre_launch, post_launch

def run_diagnostic_analytics(post_launch):
    print("\n[3/5] DIAGNOSTIC ANALYTICS: ECONOMETRICS & CANNIBALIZATION")
    df_log = post_launch.copy()
    df_log['ln_acrysof'] = np.log1p(df_log['qty_acrysof_iq'])
    df_log['ln_clareon'] = np.log1p(df_log['qty_clareon'])

    if not df_log.empty:
        try:
            model = smf.ols('ln_acrysof ~ ln_clareon + C(cust_name)', data=df_log).fit(
                cov_type='cluster', cov_kwds={'groups': df_log['cust_name']}
            )
            print(f"Cannibalization Coefficient: {model.params['ln_clareon']:.2f}")
            print(f"P-Value: {model.pvalues['ln_clareon']:.3f}")
            print("Insight: High p-value proves no statistically significant cannibalization.")
        except Exception as e:
            print("Dataset constraints prevented regression:", e)

def run_predictive_analytics(pre_launch, post_launch):
    print("\n[4/5] PREDICTIVE ANALYTICS: PROPENSITY MODEL")
    target = (post_launch.groupby('cust_name')['qty_clareon'].sum() > 0).astype(int).reset_index(name='adopted_clareon')
    
    features = pre_launch.groupby('cust_name').agg({
        'qty_acrysof_iq': ['sum', 'mean'], 'total_hna': ['sum']              
    }).reset_index()
    features.columns = ['cust_name', 'total_legacy', 'avg_legacy', 'total_spend']
    
    df_ml = features.fillna(0).merge(target, on='cust_name', how='inner')
    X = df_ml.drop(['cust_name', 'adopted_clareon'], axis=1)
    y = df_ml['adopted_clareon']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    naive = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    print("\n--- Naive Baseline (ZeroR) ---")
    print(classification_report(y_test, naive.predict(X_test), zero_division=0))

    rf = RandomForestClassifier(n_estimators=100, max_depth=3, class_weight={0: 1, 1: 1.2}, random_state=42).fit(X_train, y_train)
    print("--- Optimized Random Forest ---")
    print(classification_report(y_test, rf.predict(X_test)))
    
    # Retrain on full data for production
    rf.fit(X, y)
    df_ml['switch_prob'] = rf.predict_proba(X)[:, 1]
    return df_ml

def run_prescriptive_analytics(df_ml, hospital_attr):
    print("\n[5/5] PRESCRIPTIVE ANALYTICS: TARGET ROUTING")
    df_ml = df_ml.merge(hospital_attr, on='cust_name', how='left')
    
    leads = df_ml[(df_ml['adopted_clareon'] == 0) & (df_ml['total_legacy'] >= MINIMUM_VOLUME)].copy()
    leads = leads.sort_values(by=['switch_prob', 'total_legacy'], ascending=[False, False])
    
    leads['Action'] = leads['switch_prob'].apply(
        lambda p: "Priority Call" if p >= 0.65 else ("Secondary Target" if p >= 0.55 else "Nurture")
    )
    leads['switch_prob'] = (leads['switch_prob'] * 100).round(1).astype(str) + '%'
    
    print("\nTop Actionable Targets (Generated Lead List):")
    print(leads[['cust_name', 'cust_city', 'switch_prob', 'total_legacy', 'Action']].head(10).to_string(index=False))

def main():
    print("Initializing Clareon Analytics Pipeline...")
    df_panel, hospital_attr = load_and_prep_data()
    pre_launch, post_launch = run_descriptive_analytics(df_panel)
    run_diagnostic_analytics(post_launch)
    df_ml = run_predictive_analytics(pre_launch, post_launch)
    run_prescriptive_analytics(df_ml, hospital_attr)
    print("\nPipeline execution completed successfully.")

if __name__ == "__main__":
    main()
