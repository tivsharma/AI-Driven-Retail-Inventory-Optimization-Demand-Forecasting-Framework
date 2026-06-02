# AI-Driven-Retail-Inventory-Optimization-Demand-Forecasting-Framework

## Project Overview
This repository hosts an end-to-end predictive analytics framework designed to optimize supply chain pipelines and mitigate overstocking/understocking vulnerabilities for small-to-mid scale retail enterprises. Using a transactional dataset of 4,384 weekly records from a multi-platform supplement retail store, this project evaluates linear vs. ensemble machine learning architectures, diagnoses critical data leakage, and translates model inferences into an automated inventory recommendation engine.

## Key Metrices & Project Evolution 

### Phase 1: The Inherent Target Leakage Anomaly
* **Initial Performance:** Initial baseline models yielded deceptively near-perfect metrics (Linear Regression \(R^2 = 0.866\) | Random Forest \(R^2 = 0.977\)).
* **The Diagnostic Discovery:** Feature Importance analysis isolated a massive **Target Leakage** vector caused by the `Revenue` attribute. Because \(\text{Revenue} = \text{Units Sold} \times \text{Price}\), the models were inadvertently accessing the mathematical derivative of our target feature (`Units Sold`), creating an unrealistic forecasting environment.

### Phase 2: Refined, Real-World Econometric Engineering
Following the eradication of the leaking variables, models were retrained using purely operational and engineered rolling metrics, establishing true real-world deployment parameters:
* **Refined Linear Regression:** \(R^2 = -0.003\) (Validated that retail demand fluctuations are highly non-linear).
* **Refined Random Forest Regressor:** \(R^2 = 0.099\) | \(\text{MAE} = 9.34\).
* **XGBoost Regressor Framework:** \(R^2 = 0.087\) | \(\text{MAE} = 9.42\).

## Technical Stack & Methodology 
* **Core Languages & Libraries:** Python (Pandas, NumPy, Scikit-Learn, XGBoost).
* **Statistical Inference Engine:** SciPy Stats (Independent Sample T-Testing).
* **Data Pipelines & Feature Engineering:** Power Query, temporal extraction (Month/Week/Year), 1-step lag vectors, and 7-day rolling window averages.
* **Data Visualization:** Matplotlib, Seaborn.

## Core Analytics 

### 1. Exploratory Data Profiling & Statistical Guardrails
* **Distribution Mechanics:** Isolated a balanced, bell-shaped distribution pattern for units sold, concentrated primarily between 140 to 160 operational units.
* **Platform & Category Diagnostics:** Proved homogenous demand across distribution channels (Amazon, Walmart, iHerb) and inventory categories (Protein, Vitamins, Herbs), demonstrating zero structural skew.
* **Hypothesis Testing (Discount Impact):** Conducted an Independent Sample T-Test comparing discounted vs. non-discounted transactions. The resulting \(p\text{-value} = 0.748\) fell far above the standard 0.05 alpha threshold, statistically proving that discounts did not yield significant isolated demand shifts.

### 2. Algorithmic Inventory Actions Pipeline
The forecasting outputs were funneled into a programmatic inventory routing script to classify products and automate pipeline operations:
* **High Demand** (\(\hat{y} > 158\) units) \(\rightarrow\) **Action:** *Increase Stock* (Replenishment priority to avoid stockouts).
* **Medium Demand** (\(142 < \hat{y} \le 158\) units) \(\rightarrow\) **Action:** *Maintain Inventory* (Standard baseline hold).
* **Low Demand** (\(\hat{y} \le 142\) units) \(\rightarrow\) **Action:** *Reduce Overstock* (Mitigate capital stagnation and holding costs).

## Repository Structure 
```text
retail-inventory-optimization/
│
├── data/
│   └── supplement_retail_sales.csv   # Structured transactional dataset (4,384 rows)
│
├── notebooks/
│   ├── 1_exploratory_&_statistical_analysis.ipynb # EDA, T-Testing, & Heatmaps
│   ├── 2_leakage_diagnostics_baseline.ipynb      # Target leakage proofing scripts
│   └── 3_refined_ensemble_forecasting.ipynb      # Lag/Rolling features, RF, XGBoost
│
├── production/
│   └── inventory_recommendation_engine.py       # Algorithmic stock classification script
│
├── documentation/
│   └── AI_Based_Inventory_Management_Paper.pdf  # Full project text & academic overview
│
└── README.md                                    # Technical documentation
```

## Future Enhancement 
* Transitioning from classical tree regressors to advanced deep sequence modeling architectures (LSTM, Prophet).
* Incorporating broader supply chain variables including supplier lead times, stockout histories, and exact warehouse holding cost bounds.

## Author 
* **Tivsha Sharma**
* **Email:** ativshav25@gmail.com
* **LinkedIn:** https://www.linkedin.com/in/tivsha-sharma-3558b72ba/
