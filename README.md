#  New Product Launch: Predictive Propensity & Cannibalization Analysis
Executive Lead: Luthfia Yuwono

**Project Objective:**
To drive the commercial transition from AcrySof (Legacy) to Clareon (New Launch) by identifying high-probability hospital targets and quantifying product cannibalization.

**The Business Problem:**
The launch of a new medical device often faces two risks:
1. Sales Inertia: High-volume legacy accounts remaining stagnant.
2. Cannibalization: New product sales merely replacing legacy sales without increasing total hospital spend.
This project provides a data-driven roadmap to identify which accounts are most likely to switch and provides a Lead List of 102 stagnant accounts for the sales team to prioritize.

**Data Science Approach:**
![Analytics Pipeline Diagram](https://www.jirav.com/hs-fs/hubfs/Gartner%20Analytics.jpg?width=690&name=Gartner%20Analytics.jpg)
Moving beyond high-level business frameworks, this pipeline implements:
1. Descriptive Analytics: Categorizes hospitals based on pre- and post-launch purchasing behavior.
2. Diagnostic Analytics: A fixed-effects OLS regression to calculate cross-volume elasticity.
3. Predictive Analytics: A Random Forest Classifier compared against a Naive Baseline (Dummy Classifier) to prove statistical superiority.
4. Prescriptive Analytics: A probability-weighted lead scoring system for sales routing.

___

## Model Performance Highlights
1. The Baseline: The naive model achieved moderate accuracy by guessing "No Switch," but provided 0% Recall for actual leads.
2. The Optimized Model: Our Random Forest identifies switchers with high precision, ensuring sales resources are not wasted on false positives.
3. The Result: Regression analysis yielded an elasticity of -0.05 (p=0.514), proving that New Product (Clareon) is currently expanding the market footprint rather than cannibalizing legacy volume.

___

## O.W.L.E.T.-AI Performance Audit
This project was developed with the assistance of O.W.L.E.T.-AI. To ensure engineering integrity, a strict governance framework was applied.

Reliability: 60% of AI tasks were approved; 31% required Strategic Redirection due to context misalignment.

Strategic Redirection Example: The AI initially suggested a basic OLS regression. I overruled this in favor of a Random Forest approach to capture non-linear relationships in financial behavior that OLS would have missed.

**View the Performance Report:** [OWLET_AI_Performance-13.pdf](./OWLET_AI_Performance-13.pdf)

___

## Technical Execution
To replicate this environment, install the following dependencies:
```text
pip install -r requirements.txt
```

## Repository Structure

```text
├── 📁 .devcontainer/           # Configuration for consistent development environments
├── 📄 Final_Project.ipynb      # The sanitized, end-to-end predictive pipeline (EDA to ML)
├── 📄 OWLET_AI_Performance.pdf # Professional 1-page PDF: AI Junior Assistant Evaluation
├── 📄 OWLET_AI_Performance.py  # Python script used to generate the performance metrics/PDF
├── 📄 README.md                # Executive summary and professional project handover
├── 📄 app.py                   # Deployment script for live model interaction
└── 📄 requirements.txt         # List of dependencies (fpdf, matplotlib, sklearn, etc.)
```

