# Flight-Delay-Prediction
End-to-end flight delay prediction using AWS EMR/Hive for big data processing and Logistic Regression in Python — 180K records, 90%+ accuracy
# ✈️ Big Data Flight Delay Prediction
### IS669 Group Project — Kumari Shivani Fnu

> An end-to-end Big Data pipeline that predicts whether a US domestic flight will be delayed — using AWS cloud infrastructure for data processing and Python machine learning for classification.

---

## 📌 Project Summary

| | |
|---|---|
| **Course** | IS669 — Big Data & Cloud Computing |
| **Dataset** | Harvard Dataverse — US Airline On-Time Performance |
| **My Year** | 2007 |
| **Team Size** | 3 members (each assigned a different year) |
| **Final Dataset** | 180,000 records (60,000 per member) |
| **Goal** | Predict flight delay (Y/N) with ≥ 90% accuracy & precision |

---

## 🏗️ Pipeline Architecture

```
Harvard Dataverse (millions of rows per year)
          │
          ▼
    ┌─────────────┐
    │   AWS S3    │  ← Raw CSV data stored in cloud storage
    └─────────────┘
          │
          ▼
    ┌─────────────┐
    │  AWS EMR    │  ← Hadoop cluster (multiple EC2 instances)
    │  (Hadoop)   │    SSH access via key pair + Security Groups
    └─────────────┘
          │
          ▼
    ┌─────────────┐
    │ Apache Hive │  ← SQL-like queries to label & sample data
    │  (HiveQL)   │    Creates `shivani_sample` table, adds Delayed col
    └─────────────┘
          │  60,000 rows
          ▼
    Team combines 3 × 60,000 → 180,000 rows
          │
          ▼
    ┌──────────────────┐
    │  Google Colab    │  ← Python / pandas / scikit-learn
    │  (Jupyter)       │    Data cleaning, feature engineering, ML
    └──────────────────┘
          │
          ▼
    Logistic Regression Classifier
    → Predict: Delayed (Y=1) or On-Time (N=0)
```

---

## ☁️ Phase 1 — AWS Cloud Infrastructure

The 2007 flight dataset contains **millions of records** — far too large for local processing. The solution was a cloud-based Hadoop cluster on AWS.

### AWS Services Used

| Service | Purpose |
|---|---|
| **S3** | Stored the raw airline CSV data files |
| **EC2** | Virtual machines forming the compute cluster |
| **EMR** | Managed Hadoop + Hive installation across EC2 nodes |
| **SSH + Security Groups** | Secure terminal access from local machine to cluster |

### Steps Performed in Hive

```sql
-- 1. Create sample table with Delayed column
CREATE TABLE shivani_sample AS
SELECT *, NULL AS Delayed FROM flights_2007;

-- 2. Label flights as Delayed or Not
UPDATE shivani_sample
SET Delayed = CASE
  WHEN ArrDelay > 0 OR DepDelay > 0 THEN 'Y'
  ELSE 'N'
END;

-- 3. Random sample of 60,000 records
INSERT INTO shivani_sample
SELECT * FROM flights_2007
ORDER BY RAND()
LIMIT 60000;
```

---

## 🐍 Phase 2 — Machine Learning in Python

After all three teammates combined their labeled 60K datasets (→ 180,000 rows), the ML phase began in **Google Colab**.

### Dataset Columns

```
Year, Month, DayOfMonth, DayOfWeek, DepTime, CRSDepTime, ArrTime,
CRSArrTime, UniqueCarrier, FlightNum, TailNum, ActualElapsedTime,
CRSElapsedTime, AirTime, ArrDelay, DepDelay, Origin, Dest,
Distance, TaxiIn, TaxiOut, Cancelled, Diverted, Delayed (target)
```

### Data Preprocessing

```python
import pandas as pd
import numpy as np

# Load combined 180K dataset
df = pd.read_csv('180000Data.csv', dtype=str)

# Replace Harvard null sentinel
df.replace(r'\\N', np.nan, regex=True, inplace=True)

# Drop sparse/leaky columns
df.drop(['CancellationCode', 'CarrierDelay', 'WeatherDelay',
         'NASDelay', 'SecurityDelay', 'LateAircraftDelay'], axis=1, inplace=True)

# Convert and fill numeric columns
for col in ['DepTime', 'ArrTime', 'ArrDelay', 'DepDelay', 'Distance']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Encode target
df['Delayed'] = df['Delayed'].map({'Y': 1, 'N': 0})
```

### Model Training

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, classification_report

# Feature selection
features = ['DepTime', 'ArrTime', 'ArrDelay', 'DepDelay', 'Distance']
X = df[features]
y = df['Delayed']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))
```

---

## 📁 Repository Structure

```
flight-delay-prediction/
│
├── data/
│   ├── 180000Data.csv           # Combined 180K training dataset
│   ├── shivaniData_csv.xlsx     # 10 test flights to score
│   ├── airports.csv             # Airport reference data
│   └── carriers.csv             # Airline carrier codes
│
├── notebooks/
│   └── flight_delay_model.ipynb # Full Colab notebook
│
├── docs/
│   ├── project_requirements.docx
│   └── AWS_setup_screenshots/
│
└── README.md
```

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazon-aws&logoColor=white)
![Apache Hadoop](https://img.shields.io/badge/Hadoop-66CCFF?style=flat&logo=apache&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)

**Cloud:** AWS EMR, AWS S3, AWS EC2, SSH, Security Groups  
**Big Data:** Apache Hadoop, Apache Hive, HiveQL, Distributed Processing  
**ML/Data:** Python, pandas, numpy, scikit-learn, Logistic Regression  
**Tools:** Google Colab, Jupyter Notebook  

---

## 🎯 Key Outcomes

- ✅ Successfully configured and operated a distributed AWS EMR Hadoop cluster
- ✅ Used HiveQL to process, label, and sample from a multi-million row dataset
- ✅ Built and evaluated a binary classification model on 180,000 records
- ✅ Scored 10 real unknown test flights with Delayed (Y) / On-Time (N) prediction
- ✅ Achieved ≥ 90% accuracy and precision targets

---

## 👤 About

**Kumari Shivani Fnu** — MS Information Systems  
Project completed as part of IS669 Big Data coursework.

---

*Data sourced from the Harvard Dataverse Airline On-Time Performance dataset.*

