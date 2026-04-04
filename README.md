# 📦 Comparative Study of SVM, Random Forest & Decision Tree
### Sales Prediction and Fraud Detection · DataCo Global Supply Chain

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-red?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-CC%204.0-green)
![VCS](https://img.shields.io/badge/VCS-GitHub%20DVCS-black?logo=github)

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Team Members](#-team-members)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Machine Learning Models](#-machine-learning-models)
- [Results Summary](#-results-summary)
- [Getting Started](#-getting-started)
- [Version Control Workflow](#-version-control-workflow)
- [Dependencies](#-dependencies)
- [Key Findings](#-key-findings)
- [Conclusion](#-conclusion)

---

## 🔍 Project Overview

This project conducts a **comparative study** of three popular supervised machine learning classifiers applied to a real-world supply chain dataset from **DataCo Global**. The study evaluates each model's ability to:

1. **Detect fraudulent transactions** — identifying suspected fraud orders (binary classification)
2. **Predict late deliveries** — flagging orders at risk of delayed shipment (binary classification)

The project was developed collaboratively using **GitHub as a Distributed Version Control System (DVCS)**, with each team member owning a dedicated branch and model. Models were merged via Pull Requests after peer code review.

> **Course:** Collaborative Software Development  
> **Assessment:** Practical — Local VCS / Distributed VCS  
> **Dataset:** DataCo Global Supply Chain (~180,000 transactions)

---

## 👥 Team Members

| Name | Model Implemented | Branch | Role |
|------|-------------------|--------|------|
| **Omar** | Support Vector Machine (SVM) | `main` | SVM Classification |
| **Sammy** | Random Forest Classifier | `randomforest` | Ensemble Classification |
| **Terence** | Decision Tree Classifier | `decisiontree` | Tree-based Classification |

---

## 📁 Repository Structure

```
├── comparison_of_classification_regression_rnn.ipynb   # Main notebook (merged)
├── README.md                                            # This file
└── DataCoSupplyChainDataset.csv                        # Dataset (download separately)
```

> **Note:** The dataset must be downloaded separately from [Mendeley Data](https://data.mendeley.com/datasets/8gx2fvg2k6/5) and placed in the project root directory.

---

## 📊 Dataset

| Attribute | Detail |
|-----------|--------|
| **Source** | Mendeley Data — Fabian Constante, Fernando Silva, António Pereira |
| **Licence** | Creative Commons 4.0 |
| **Records** | ~180,519 transactions |
| **Features** | 53 columns (reduced to 41 after preprocessing) |
| **Period** | 3 years of supply chain operations |
| **Link** | https://data.mendeley.com/datasets/8gx2fvg2k6/5 |

### Key Features Used

- `Order Status` → binary fraud label (`1` = SUSPECTED_FRAUD)
- `Delivery Status` → binary late delivery label (`1` = Late delivery)
- `Late_delivery_risk`, `Shipping Mode`, `Order Region`, `Category Name`
- `Product Price`, `Sales per customer`, `Benefit per order`
- `Type` (payment method), `Market`, `Customer Segment`

### Preprocessing Steps

1. Dropped irrelevant columns (emails, passwords, images, coordinates)
2. Filled 3 missing `Customer Zipcode` values with `0`
3. Engineered `Customer Full Name` from first + last name
4. Created binary target columns: `fraud` and `late_delivery`
5. Dropped leakage columns: `Delivery Status`, `Late_delivery_risk`, `Order Status`
6. Label-encoded 16 categorical (object-type) columns
7. Applied `StandardScaler` to all feature sets (fit on train, transform on test)
8. Split: **80% training / 20% testing** (`random_state=42`)

---

## 🤖 Machine Learning Models

### 1. Support Vector Machine — Omar (`main`)

```python
from sklearn import svm

model_f = svm.LinearSVC()
model_l = svm.LinearSVC()
```

**How it works:** LinearSVC finds an optimal hyperplane that maximises the margin between classes in a high-dimensional feature space. Effective for linearly separable problems with many features.

---

### 2. Random Forest Classifier — Sammy (`randomforest`)

```python
from sklearn.ensemble import RandomForestClassifier

model_f = RandomForestClassifier()
model_l = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
```

**How it works:** Builds multiple decision trees on random subsets of data and features, then aggregates predictions via majority voting — reducing overfitting and improving generalisation.

---

### 3. Decision Tree Classifier — Terence (`decisiontree`)

```python
from sklearn import tree

model_f = tree.DecisionTreeClassifier()
model_l = tree.DecisionTreeClassifier()
```

**How it works:** Recursively partitions the feature space using the Gini impurity criterion, producing interpretable decision rules. Each split maximises class separation.

---

## 📈 Results Summary

### Classification Performance (%)

| Model | Fraud — Accuracy | Fraud — Recall | Fraud — F1 | Late Delivery — Accuracy | Late Delivery — Recall | Late Delivery — F1 |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| SVM (Omar) | 97.75 | 56.89 | 28.40 | **98.84** | 97.94 | 98.96 |
| Random Forest (Sammy) | 98.68 | **97.70** | 61.56 | 98.61 | 97.52 | 98.75 |
| **Decision Tree (Terence)** | **99.09** | 81.91 | **80.34** | **99.37** | **99.43** | **99.42** |

> ⭐ **Best scores in bold.** F1-score is the primary evaluation metric as it balances precision and recall — especially important for imbalanced fraud data.

### Cross-Validation

6-fold cross-validation was applied to confirm results. The difference between cross-validated accuracy and held-out test accuracy was minimal for all models, confirming that no model was overfitted.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab
- Git

### Installation

```bash
# 1. Clone this repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# 2. Install required packages
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras xgboost lightgbm statsmodels

# 3. Download the dataset
# Visit: https://data.mendeley.com/datasets/8gx2fvg2k6/5
# Place DataCoSupplyChainDataset.csv in the project root

# 4. Open the notebook
jupyter notebook comparison_of_classification_regression_rnn.ipynb
```

### Running the Models

The notebook is structured sequentially. Run all cells from top to bottom:

1. **Cells 1–6** — Library imports and dataset loading
2. **Cells 7–20** — Data cleaning and visualisation
3. **Cells 21–100** — EDA, customer segmentation (RFM analysis)
4. **Cells 101–112** — Data modelling (train/test split, scaling)
5. **Cells 107–108** — SVM model (Omar)
6. **Cells 109–110** — Random Forest model (Sammy)
7. **Cells 111–112** — Decision Tree model (Terence)
8. **Cells 113–122** — Comparison table and cross-validation

> **Important:** Update the dataset path in Cell 7 to match your local file location:
> ```python
> dataset = pd.read_csv(r"DataCoSupplyChainDataset.csv", encoding='unicode_escape')
> ```

---

## 🔀 Version Control Workflow

This project was developed using **GitHub (Distributed VCS)**. The workflow is illustrated below:

```
Lecturer's Repo
      │
      ▼
   git clone  (×3 students)
      │
      ├── Omar  ──────────────►  main branch       (SVM model)
      │
      ├── Sammy ──────────────►  randomforest branch  (Random Forest)
      │
      └── Terence ────────────►  decisiontree branch  (Decision Tree)
                                         │
                               git push (each branch)
                                         │
                               Pull Requests on GitHub
                                         │
                               Code Review & Approval
                                         │
                               Merge into main ✅
```

### Key Git Commands Used

```bash
# Clone the repository
git clone <repository-url>

# Create and switch to a feature branch (Sammy & Terence)
git checkout -b randomforest
git checkout -b decisiontree

# Stage and commit changes
git add .
git commit -m "Add Random Forest classifier for fraud and late delivery detection"

# Push branch to remote
git push origin randomforest

# After PR review and merge (on GitHub web interface)
git checkout main
git pull origin main
```

---

## 📦 Dependencies

```txt
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
keras
xgboost
lightgbm
statsmodels
```

Install all at once:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras xgboost lightgbm statsmodels
```

---

## 💡 Key Findings

### Fraud Detection
- **SVM performed poorly** — despite high accuracy (97.75%), its recall of only 56.89% means it missed nearly half of all actual fraud cases. Its F1-score of 28.40% reflects this critically.
- **Random Forest had the best recall** (97.70%) — it caught almost all fraud cases but at lower precision, giving an F1 of 61.56%.
- **Decision Tree was the overall winner** — F1-score of 80.34%, the best balance of precision and recall for fraud detection.

### Late Delivery Prediction
- **All three models excelled** — F1-scores above 98%, suggesting late delivery is highly predictable from the available features.
- The dataset confirms that **all orders flagged with `late_delivery_risk = 1` were actually delivered late** — this strong signal makes the task more learnable.
- **Decision Tree led** with 99.42% F1 and 99.43% recall.

### Dataset Insights
- 🌍 **Western Europe** had the highest sales and the highest fraud rate (~17.4% of all suspected fraud)
- 👟 **Cleats** category had the most late deliveries and most fraud instances
- 💳 **All fraud orders** were placed using wire transfer (TRANSFER type) — no fraud via Debit, Cash, or Payment
- 👤 **Mary Smith** alone attempted 528 fraudulent transactions totalling ~$102,000
- 📅 Sales were consistent from Q1 2015 to Q3 2017, then dropped ~65% in Q1 2018

---

## ✅ Conclusion

The **Decision Tree Classifier** demonstrated the strongest overall performance across both classification tasks, making it the recommended model for this dataset. For fraud detection specifically, it significantly outperformed SVM (F1: 80.34% vs 28.40%), while remaining competitive on late delivery prediction.

The use of **GitHub as a DVCS** enabled smooth parallel development across three team members. Branching ensured isolated workspaces, Pull Requests enforced code review, and merging consolidated all contributions into a single, unified notebook. This mirrors industry-standard collaborative development practices.

---

## 📄 Licence

Dataset: [Creative Commons 4.0](https://creativecommons.org/licenses/by/4.0/) — Fabian Constante, Fernando Silva, António Pereira via Mendeley Data.

---

*Collaborative Software Development — Practical Assessment · DataCo Global Supply Chain Project*
