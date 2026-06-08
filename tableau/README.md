# Tableau Dashboard Blueprint

This folder contains the source data for Tableau dashboards and the workbook design guidance.

## Data Source

Use `tableau/dashboard_data/predictions_enriched.csv` as the Tableau data source.

A Python dashboard alternative is available at `tableau/dashboard_app.py`. Run it to generate `tableau/dashboard.html` from the enriched prediction data.

### Columns available
- `transaction_id`
- `amount`
- `fraud_probability`
- `prediction`
- `actual`
- `prediction_label`
- `actual_label`
- `risk_category`

---

## Dashboard 1: Executive Overview

### KPIs

1. Total Transactions
   - Measure: `COUNT([transaction_id])`

2. Fraud Predictions
   - Measure: `SUM([prediction])`
   - Since `prediction` is `1` for fraud, this gives fraud count.

3. Fraud Rate
   - Calculated Field: `SUM([prediction]) / COUNT([transaction_id])`
   - Format as Percentage.

4. Fraud Amount
   - Calculated Field:
     ```text
     IF [prediction] = 1 THEN [amount] END
     ```
   - KPI: `SUM([Fraud Amount])`

5. Average Fraud Probability
   - Measure: `AVG([fraud_probability])`

---

## Dashboard 2: Fraud Investigation

### Risk Category Distribution

- Use `risk_category` on Rows
- Use `COUNT([transaction_id])` on Columns
- This shows transactions by risk category.

### Top Fraud Probability Transactions

- Apply filter: `prediction = 1`
- Sort by: `fraud_probability` descending
- Display columns:
  - `transaction_id`
  - `amount`
  - `fraud_probability`
  - `actual`

### Fraud Amount by Risk Category

- Rows: `risk_category`
- Columns: `SUM([amount])`
- This shows where money is at risk.

---

## Dashboard 3: Model Performance

### Confusion Matrix Metrics

Create the following calculated fields:

- `True Positive`
  ```text
  IF [prediction] = 1 AND [actual] = 1 THEN 1 END
  ```

- `False Positive`
  ```text
  IF [prediction] = 1 AND [actual] = 0 THEN 1 END
  ```

- `True Negative`
  ```text
  IF [prediction] = 0 AND [actual] = 0 THEN 1 END
  ```

- `False Negative`
  ```text
  IF [prediction] = 0 AND [actual] = 1 THEN 1 END
  ```

### Precision

```text
SUM([True Positive]) / (SUM([True Positive]) + SUM([False Positive]))
```

### Recall

```text
SUM([True Positive]) / (SUM([True Positive]) + SUM([False Negative]))
```

### F1 Score

You can calculate this in Python and import it, or use:

```text
2 * SUM([True Positive]) /
(
  2 * SUM([True Positive]) + SUM([False Positive]) + SUM([False Negative])
)
```

---

## Dashboard 4: Probability Analysis

### Probability Histogram

- Create bins for `fraud_probability` with size `0.05`.
- Plot `COUNT([transaction_id])` against `fraud_probability (bin)`.

### Fraud Probability vs Amount

- Columns: `fraud_probability`
- Rows: `amount`
- Color: `prediction`
- Marks: Circle

---

## Dashboard 5: Financial Impact

### Fraud Amount Detected

```text
SUM(IF [prediction] = 1 THEN [amount] END)
```

### Missed Fraud Amount

```text
SUM(IF [prediction] = 0 AND [actual] = 1 THEN [amount] END)
```

### Correctly Prevented Fraud

```text
SUM(IF [prediction] = 1 AND [actual] = 1 THEN [amount] END)
```

---

## Notes

- Use `predictions_enriched.csv` as the single source for all dashboards.
- If you want to refresh the data, re-run `notebooks/02_modeling.ipynb` to regenerate the CSV files.
- `risk_category` is already pre-binned into `Low`, `Medium`, `High`, and `Critical`.
