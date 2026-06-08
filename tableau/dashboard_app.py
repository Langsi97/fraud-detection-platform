from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.io import write_html
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DATA_FILE = Path(__file__).parent / "dashboard_data" / "predictions_enriched.csv"
OUTPUT_FILE = Path(__file__).parent / "dashboard.html"

RISK_ORDER = ["Low", "Medium", "High", "Critical"]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "risk_category" in df.columns:
        df["risk_category"] = pd.Categorical(df["risk_category"], categories=RISK_ORDER, ordered=True)
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["actual"].astype(int)
    y_pred = df["prediction"].astype(int)
    fraud_probability = df["fraud_probability"].astype(float)

    metrics = {
        "total_transactions": int(df["transaction_id"].nunique()),
        "predicted_fraud_count": int(y_pred.sum()),
        "actual_fraud_count": int(y_true.sum()),
        "fraud_rate_predicted": float(y_pred.mean()),
        "fraud_rate_actual": float(y_true.mean()),
        "average_probability": float(fraud_probability.mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, fraud_probability))
    except Exception:
        metrics["roc_auc"] = None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update({
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    })

    return metrics


def format_pct(value: float) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def build_metrics_html(metrics: dict) -> str:
    roc_auc_display = f"{metrics['roc_auc']:.4f}" if metrics['roc_auc'] is not None else "n/a"
    return f"""
    <section>
        <h2>Executive Overview</h2>
        <div class='metrics-grid'>
            <div class='metric-card'><span class='metric-value'>{metrics['total_transactions']:,}</span><span class='metric-label'>Total Transactions</span></div>
            <div class='metric-card'><span class='metric-value'>{metrics['predicted_fraud_count']:,}</span><span class='metric-label'>Predicted Fraud</span></div>
            <div class='metric-card'><span class='metric-value'>{format_pct(metrics['fraud_rate_predicted'])}</span><span class='metric-label'>Predicted Fraud Rate</span></div>
            <div class='metric-card'><span class='metric-value'>{metrics['actual_fraud_count']:,}</span><span class='metric-label'>Actual Fraud Cases</span></div>
            <div class='metric-card'><span class='metric-value'>{format_pct(metrics['fraud_rate_actual'])}</span><span class='metric-label'>Actual Fraud Rate</span></div>
            <div class='metric-card'><span class='metric-value'>{metrics['average_probability']:.4f}</span><span class='metric-label'>Average Fraud Probability</span></div>
            <div class='metric-card'><span class='metric-value'>{metrics['accuracy']:.4f}</span><span class='metric-label'>Accuracy</span></div>
            <div class='metric-card'><span class='metric-value'>{metrics['precision']:.4f}</span><span class='metric-label'>Precision</span></div>
            <div class='metric-card'><span class='metric-value'>{metrics['recall']:.4f}</span><span class='metric-label'>Recall</span></div>
            <div class='metric-card'><span class='metric-value'>{metrics['f1_score']:.4f}</span><span class='metric-label'>F1 Score</span></div>
            <div class='metric-card'><span class='metric-value'>{roc_auc_display}</span><span class='metric-label'>ROC AUC</span></div>
        </div>
    </section>
    """


def create_risk_category_figure(df: pd.DataFrame) -> go.Figure:
    counts = df["risk_category"].value_counts().reindex(RISK_ORDER).fillna(0)
    return go.Figure(
        data=[go.Bar(x=counts.index, y=counts.values, marker_color="#636efa")],
        layout=go.Layout(
            title="Transactions by Risk Category",
            xaxis_title="Risk Category",
            yaxis_title="Transaction Count",
        ),
    )


def create_amount_by_risk_figure(df: pd.DataFrame) -> go.Figure:
    sums = df.groupby("risk_category")["amount"].sum().reindex(RISK_ORDER).fillna(0)
    return go.Figure(
        data=[go.Bar(x=sums.index, y=sums.values, marker_color="#ef553b")],
        layout=go.Layout(
            title="Total Transaction Amount by Risk Category",
            xaxis_title="Risk Category",
            yaxis_title="Total Amount",
        ),
    )


def create_probability_histogram(df: pd.DataFrame) -> go.Figure:
    return go.Figure(
        data=[
            go.Histogram(
                x=df["fraud_probability"],
                nbinsx=30,
                marker_color="#00cc96",
                opacity=0.8,
            )
        ],
        layout=go.Layout(
            title="Fraud Probability Distribution",
            xaxis_title="Fraud Probability",
            yaxis_title="Transaction Count",
        ),
    )


def create_confusion_matrix_figure(df: pd.DataFrame) -> go.Figure:
    labels = ["Legitimate", "Fraud"]
    y_true = df["actual"].astype(int)
    y_pred = df["prediction"].astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return go.Figure(
        data=[
            go.Heatmap(
                z=matrix,
                x=labels,
                y=labels,
                colorscale="Blues",
                hovertemplate="Predicted %{x}<br>Actual %{y}<br>Count %{z}<extra></extra>",
            )
        ],
        layout=go.Layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
        ),
    )


def create_amount_probability_scatter(df: pd.DataFrame) -> go.Figure:
    return go.Figure(
        data=[
            go.Scatter(
                x=df["fraud_probability"],
                y=df["amount"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=df["prediction"],
                    colorscale=[[0, "#636efa"], [1, "#ef553b"]],
                    colorbar=dict(title="Prediction"),
                    showscale=True,
                ),
                text=df["risk_category"],
                hovertemplate="Transaction %{customdata[0]}<br>Amount %{y:$,.2f}<br>Probability %{x:.4f}<br>Risk %{text}<extra></extra>",
                customdata=df[["transaction_id"]].values,
            )
        ],
        layout=go.Layout(
            title="Amount vs. Fraud Probability",
            xaxis_title="Fraud Probability",
            yaxis_title="Amount",
        ),
    )


def create_top_fraud_table(df: pd.DataFrame, max_rows: int = 20) -> go.Figure:
    top_fraud = (
        df[df["prediction"] == 1]
        .sort_values("fraud_probability", ascending=False)
        .head(max_rows)
        .loc[:, ["transaction_id", "amount", "fraud_probability", "actual_label", "risk_category"]]
    )
    return go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Transaction ID", "Amount", "Fraud Probability", "Actual Label", "Risk Category"],
                    fill_color="#2a3f5f",
                    font=dict(color="white", size=12),
                ),
                cells=dict(
                    values=[
                        top_fraud["transaction_id"],
                        top_fraud["amount"].map("${:,.2f}".format),
                        top_fraud["fraud_probability"].map("{:.6f}".format),
                        top_fraud["actual_label"],
                        top_fraud["risk_category"],
                    ],
                    fill_color="#f5f5f5",
                ),
            )
        ],
        layout=go.Layout(title=f"Top {len(top_fraud)} Predicted Fraud Transactions"),
    )


def build_dashboard(df: pd.DataFrame, metrics: dict, output_path: Path) -> None:
    figures = [
        create_risk_category_figure(df),
        create_amount_by_risk_figure(df),
        create_probability_histogram(df),
        create_confusion_matrix_figure(df),
        create_amount_probability_scatter(df),
        create_top_fraud_table(df),
    ]

    html_content = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
        <meta charset=\"UTF-8\" />
        <title>Fraud Detection Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 24px; background: #f8f9fb; }}
            h1, h2 {{ color: #1f2937; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 16px 0 32px; }}
            .metric-card {{ background: white; border: 1px solid #e5e7eb; border-radius: 12px; padding: 18px; box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08); }}
            .metric-value {{ display: block; font-size: 1.6rem; font-weight: 700; color: #111827; }}
            .metric-label {{ color: #6b7280; margin-top: 8px; display: block; }}
            section {{ margin-bottom: 32px; }}
            .section-header {{ margin-bottom: 8px; }}
            .figure-block {{ margin-bottom: 40px; }}
        </style>
    </head>
    <body>
        <h1>Fraud Detection Dashboard</h1>
        <p>Dashboard generated from <code>tableau/dashboard_data/predictions_enriched.csv</code>.</p>
        {build_metrics_html(metrics)}
    """

    for fig in figures:
        html_content += "<div class='figure-block'>"
        html_content += fig.to_html(full_html=False, include_plotlyjs='cdn')
        html_content += "</div>"

    html_content += "</body></html>"
    output_path.write_text(html_content, encoding="utf-8")
    print(f"Saved dashboard to: {output_path}")


if __name__ == "__main__":
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Can not find the data file: {DATA_FILE}")

    df = load_data(DATA_FILE)
    metrics = compute_metrics(df)
    build_dashboard(df, metrics, OUTPUT_FILE)
