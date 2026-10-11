from pathlib import Path

content = r"""# MACHINE LEARNING ASSIGNMENT

**Task:** Task 13 – Anomaly Detection in Credit Card Transactions  
**Application:** Credit Card Fraud Detection  
**Algorithm:** DBSCAN (Density-Based Spatial Clustering of Applications with Noise)  
**Type:** Unsupervised Machine Learning

## 1. Problem Statement

The goal of this project is to find unusual credit card transactions that may be fraudulent.

For example, a very large payment made late at night from an unknown location may be suspicious. The system looks for transactions that are different from common spending patterns.

This is **unsupervised learning** because the algorithm finds patterns without needing every transaction to be labelled as normal or fraud in advance.

## 2. Dataset Description

The sample dataset contains credit card transaction details:

- **Amount:** How much money was spent.
- **Time:** When the transaction happened.
- **Location:** Where the transaction happened.

Example transactions:

| Amount | Time | Location | Possible meaning |
|---:|---|---|---|
| $50 | 2 PM | Home city | Looks normal |
| $5,000 | 3 AM | Foreign country | May be suspicious |
| $75 | 1 PM | Home city | Looks normal |
| $10,000 | 4 AM | Unknown location | May be suspicious |

Before applying the algorithm, the data is cleaned. Numerical features are scaled, and location values are converted into numerical codes so the algorithm can process them.

## 3. Methodology

### Algorithm: DBSCAN

DBSCAN groups data points that are close to one another in a dense area. Points that do not belong to a dense group can be marked as **noise**. In this project, noise points are treated as possible anomalies.

### Steps

1. Collect sample credit card transaction data.
2. Handle missing or incorrect values.
3. Scale the transaction amount and time so large numerical values do not dominate.
4. Convert location categories into numerical values.
5. Run DBSCAN on the prepared data.
6. Mark points with cluster label `-1` as possible anomalies.
7. Review the flagged transactions.

**Important:** An anomaly is not automatically confirmed fraud. A bank would need more checks before blocking a transaction.

## 4. Python Code

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# Sample transaction data
transactions = pd.DataFrame({
    "Amount": [50, 5000, 75, 100, 10000, 60, 80, 4500, 90, 5500],
    "Time": [14, 3, 13, 14, 4, 15, 12, 2, 14, 5],
    "Location_Code": [1, 2, 1, 1, 3, 1, 1, 2, 1, 3]
})

# Scale the features
scaler = StandardScaler()
X = scaler.fit_transform(transactions)

# Find groups and isolated points
model = DBSCAN(eps=0.5, min_samples=3)
transactions["Cluster"] = model.fit_predict(X)

# DBSCAN uses -1 for points marked as noise
transactions["Anomaly"] = transactions["Cluster"] == -1

print(transactions)
print("\\nPossible anomalies:")
print(transactions[transactions["Anomaly"]])
```

### Understanding the code

- `StandardScaler()` puts numerical features on a comparable scale.
- `DBSCAN(eps=0.5, min_samples=3)` searches for dense groups using the chosen settings.
- `fit_predict(X)` assigns a cluster label to each transaction.
- A cluster label of `-1` means DBSCAN treated that point as noise.
- The `Anomaly` column is `True` for those noise points.

The `eps` and `min_samples` settings affect the results. They may need adjustment for a real dataset.

## 5. Expected Results

The assignment uses these patterns to explain the result:

- **Likely normal:** smaller amounts during the day from a familiar location.
- **Potentially unusual:** very large amounts during the early morning from a foreign or unknown location.

The original example reports 5 normal transactions and 4 anomalous transactions out of 10. These are illustrative figures; the exact output must be checked by running the code. The listed counts do not account for all 10 sample transactions, so do not treat them as a guaranteed model result.

## 6. Conclusion

This project shows how DBSCAN can help find unusual credit card transactions without using pre-labelled fraud examples. It groups transactions with similar features and marks isolated transactions as possible anomalies.

### Key learnings

1. Transaction amount, time, and location can help describe spending patterns.
2. Scaling features is important because they use different numerical ranges.
3. DBSCAN does not require the number of clusters to be set in advance.
4. A `-1` label means noise, not proven fraud.
5. Real fraud detection needs suitable data, careful testing, and additional checks.

### Real-world applications

- Credit card fraud monitoring
- Banking security
- Insurance claim review
- Cybersecurity anomaly detection

---

**End of Assignment**
"""

out = Path("/mnt/data/TASK13_SIMPLE_EXPLANATION.md")
out.write_text(content, encoding="utf-8")
print(f"Created: {out}")
