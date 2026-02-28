# Bridge Condition Prediction: Machine Learning for Infrastructure Safety

**Team 03:** Vashishth Doshi & Priyal Shah  
**Course:** 95-885 Data Science and Big Data, Fall 2025  


## Executive Summary
The United States maintains over 617,000 bridges, with approximately 42,000 classified as "Poor" condition. These represent significant public safety risks and potential economic losses exceeding $1 billion per bridge failure due to collapse costs, lawsuits, and infrastructure disruption. This project develops a machine learning system to predict Poor-condition bridges using the Federal Highway Administration's National Bridge Inventory (NBI) dataset, enabling proactive inspection prioritization and optimal resource allocation.

**Key Findings:**
- Achieved **87% recall** in identifying Poor-condition bridges using Random Forest and Gradient Boosting models.
- Quantified annual operational costs at **USD 6.18 billion** based on current missed inspections and false alarms, with a demonstrated path to reducing costs to ~$3 billion annually via threshold tuning.
- Identified critical data leakage risks (target-correlated features) and implemented robust temporal validation (training on 2023-2024 data, testing on 2025 data).
- Evaluated 17 machine learning models across 4 distinct dataset configurations to ensure rigorous model selection.

## Dataset
**Federal Highway Administration (FHWA) National Bridge Inventory (NBI)**
- **Scope:** 1,868,991 records covering the years 2023–2025, containing 124 structural, operational, and administrative features.
- **Target Variable:** `BRIDGE_CONDITION` (Good, Fair, Poor). The dataset is highly imbalanced with "Poor" bridges representing only 6.79% of the training population.
- **Data Splits:** Models were trained on 2023 and 2024 data to prevent future data leakage, and tested strictly on 2025 data to simulate real-world future predictions.

## Repository Structure
To replicate this project and its environment, structure your repository as follows:

```text
├── notebooks/
│   ├── 01_Final_Report.ipynb             # Project executive summary and business value
│   ├── 02_Data_Cleaning_and_Preprocessing.ipynb  # Missing value imputation, leakage removal
│   ├── 02b_Feature_Engineering.ipynb     # Feature creation and extraction
│   ├── 03_Class_Imbalance_Handling.ipynb # SMOTE and class-balancing techniques
│   ├── 04_Principal_Component_Analysis.ipynb # Dimensionality reduction (130 -> optimal subset)
│   ├── 05_Baseline_Models.ipynb          # LogReg, Decision Tree, Naive Bayes, KNN
│   ├── 06_Advanced_Models_RF_XGB.ipynb   # Random Forest and XGBoost training
│   ├── 07_Advanced_Models.ipynb          # Neural Networks (MLP) and HistGradientBoosting
│   ├── 08_Model_Evaluations.ipynb        # High-level evaluation of all 17 models
│   └── 09_Deeper_Comparative_Analysis.ipynb # Deep dive into the Top 3 performing models
├── models/                               # (.pkl) trained models
├── results/                              # Evaluation metrics, CSVs, and visualizations
└── README.md                             # Project overview and instructions
```

## Methodology

1. **Data Preprocessing & Leakage Removal:** Conducted missing value imputation (median for numeric, mode for categorical) and robust scaling. Crucially, removed 5 features (e.g., `LOWEST_RATING`, `DECK_COND_058`) that acted as direct proxies for the target variable, preventing artificial performance inflation.
2. **Feature Engineering & PCA:** Generated specialized features based on geometry, traffic, and inspection history. Applied Principal Component Analysis (PCA) to evaluate dimensionality reduction.
3. **Class Imbalance Handling:** Addressed the extreme 14:1 class imbalance using SMOTE (Synthetic Minority Over-sampling Technique) and calculated class weights to prioritize the detection of the minority "Poor" class.
4. **Modeling:** Trained and evaluated baseline models alongside advanced ensemble techniques (Random Forest, XGBoost, HistGradientBoosting) and Neural Networks. 
5. **Evaluation:** Focused heavily on **Recall** as the primary business metric to minimize false negatives (missed failing bridges = safety gaps), alongside precision, F1-score, and translated operational cost projections. 

## Best Performing Models

A deeper comparative analysis (`Notebook 09`) identified the following top 3 models:
1. **HistGradient Boosting (Original Dataset):** 87.0% Recall (Poor), 47.0% Precision, 74.9% Accuracy
2. **Random Forest (Original Dataset - 102 features):** 87.0% Recall (Poor), 48.1% Precision, 75.6% Accuracy 
3. **Random Forest (Engineered Dataset - 130 features):** 87.0% Recall (Poor), 48.4% Precision, 75.4% Accuracy

## Datasets available here - 
https://www.fhwa.dot.gov/bridge/nbi/ascii.cfm
