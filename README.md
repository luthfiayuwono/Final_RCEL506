# 🚀 Clareon Launch Analytics Suite: Market Expansion & Lead Prediction Pipeline

## 📊 Executive Summary
**The Business Problem:** Following the launch of the new Clareon product line, the business faced two critical strategic questions: Is Clareon cannibalizing our legacy AcrySof volume (Threat of Substitution), and how do we efficiently target the 102 stagnant hospitals that have yet to adopt the new technology (Bargaining Power of Buyers)? 

**The Solution:** This project replaces theoretical business frameworks with an automated, end-to-end data science pipeline. The suite utilizes diagnostic econometrics to mathematically prove that the new product launch is expanding market share rather than cannibalizing legacy sales. It then deploys a robust Machine Learning propensity model (Random Forest) to predict future adopters, outperforming a naive baseline by guaranteeing high precision. Finally, it translates these predictive probabilities into a prescriptive, routed "Hit List" for the sales team, maximizing operational efficiency.

---

## 🔬 The Analytics Methodology
![Analytics Pipeline Diagram](https://www.jirav.com/hs-fs/hubfs/Gartner%20Analytics.jpg?width=690&name=Gartner%20Analytics.jpg)
This pipeline takes stakeholders through the complete analytics maturity curve:

1. **Descriptive Analytics (Financial Segmentation):** Categorizes hospitals into actionable cohorts (Stagnant, Cannibalizer, New Market, True Growth) based on pre- and post-launch purchasing behavior.
2. **Diagnostic Analytics (Econometrics & Cannibalization):** Utilizes a `statsmodels` Fixed-Effects OLS Regression with clustered standard errors to isolate the cross-volume elasticity between Clareon and AcrySof. 
   * *Business Insight:* The model returned a coefficient of -0.05 with a p-value of 0.514, proving zero statistically significant cannibalization.
3. **Predictive Analytics (Propensity Modeling):** Deploys a Random Forest Classifier to identify hospitals highly likely to switch to Clareon.
   * *Model Superiority:* The model is explicitly evaluated against a `DummyClassifier` (Naive Baseline). While the baseline achieves moderate accuracy due to class imbalance by guessing "No Switch" for everyone, the optimized Random Forest sacrifices baseline recall for high Precision, ensuring sales reps do not waste time on false positives.
4. **Prescriptive Analytics (Strategic Lead Routing):** Applies deterministic business logic to the ML probabilities, filtering for minimum historical volume (>=10 units) and assigning categorical sales actions (e.g., "Immediate Priority Call") to generate a final top-10 hit list.

---

## 📂 Repository Structure

```text
├── 📄 main_pipeline.ipynb          # The sanitized, executable Jupyter Notebook
├── 📄 OWLET_AI_Performance.pdf     # One-page evaluation of AI Junior Assistant
├── 📄 requirements.txt             # Python dependencies required to run the code
└── 📄 README.md                    # Executive summary and replication instructions
