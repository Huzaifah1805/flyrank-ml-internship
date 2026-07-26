"""ML-04 Data Contract - execution script to verify notebook logic runs cleanly."""
import sys, json
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("Installing duckdb..."); import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'duckdb'])
    import duckdb

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
RAW  = ROOT / 'data/raw/content_refresh_anonymized.csv'
OUT  = ROOT / 'work/outputs'
OUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()
con.execute(f"""
    CREATE TABLE content_refresh AS
    SELECT *, '2026-03' AS month
    FROM read_csv_auto('{RAW.as_posix()}', header=True)
""")
print(f"Table loaded: {con.execute('SELECT COUNT(*) FROM content_refresh').fetchone()[0]:,} rows")

# Query 1 - Grain
q1 = con.execute("""
    SELECT COUNT(*) AS total_rows, COUNT(DISTINCT content_id) AS unique_content_ids,
           COUNT(DISTINCT client_id) AS unique_clients,
           (COUNT(*) = COUNT(DISTINCT content_id)) AS grain_is_content_id
    FROM content_refresh WHERE month = '2026-03'
""").df()
print("\nQuery 1 — Grain Check:"); print(q1.to_string(index=False))

# Query 2 - Row count + span
q2 = con.execute("""
    SELECT month, COUNT(*) AS row_count, MIN(content_age_days) AS min_age,
           MAX(content_age_days) AS max_age, ROUND(AVG(content_age_days),1) AS avg_age,
           COUNT(CASE WHEN trend_direction='down' THEN 1 END) AS declining,
           COUNT(CASE WHEN trend_direction='stable' THEN 1 END) AS stable,
           COUNT(CASE WHEN trend_direction='up' THEN 1 END) AS improving
    FROM content_refresh WHERE month = '2026-03' GROUP BY month
""").df()
print("\nQuery 2 — Row Count & Span:"); print(q2.to_string(index=False))

# Query 3 - Availability IS TRUE
q3 = con.execute("""
    SELECT COUNT(*) AS total_rows,
           COUNT(CASE WHEN (impressions_90d > 0) IS TRUE THEN 1 END) AS has_impressions,
           COUNT(CASE WHEN (impressions_90d > 0) IS TRUE AND (sessions_90d > 0) IS TRUE THEN 1 END) AS has_both,
           ROUND(100.0*COUNT(CASE WHEN (impressions_90d>0) IS TRUE THEN 1 END)/COUNT(*),1) AS pct_available
    FROM content_refresh WHERE month = '2026-03'
""").df()
print("\nQuery 3 — Availability IS TRUE:"); print(q3.to_string(index=False))

# Feature frame + leakage trap
data = con.execute("""
    SELECT days_since_last_update AS f_staleness_days,
           ctr / (avg_position + 1.0) AS f_ctr_per_position,
           CAST(impressions_last_30d AS DOUBLE)/(impressions_prev_30d+1.0) AS f_impression_momentum,
           CAST(engaged_sessions_90d AS DOUBLE)/(sessions_90d+1.0) AS f_engagement_rate,
           CAST(ai_sessions_90d AS DOUBLE)/(sessions_90d+1.0) AS f_ai_traffic_ratio,
           COALESCE(trend_pct, 0) AS LEAKED_trend_pct,
           (trend_direction='down')::INTEGER AS label
    FROM content_refresh WHERE month='2026-03' AND (impressions_90d>0) IS TRUE
""").df().fillna(0)

honest_cols = ['f_staleness_days','f_ctr_per_position','f_impression_momentum','f_engagement_rate','f_ai_traffic_ratio']
pipe = Pipeline([('scaler',StandardScaler()),('clf',LogisticRegression(max_iter=500,random_state=42))])

score_honest = cross_val_score(pipe, data[honest_cols], data['label'], cv=5, scoring='roc_auc').mean()
score_leaked = cross_val_score(pipe, data[honest_cols+['LEAKED_trend_pct']], data['label'], cv=5, scoring='roc_auc').mean()

print(f"\n=== LEAKAGE TRAP ===")
print(f"ROC-AUC honest features: {score_honest:.4f}")
print(f"ROC-AUC with trend_pct:  {score_leaked:.4f}  <-- label leakage!")
print(f"Lift:                    +{score_leaked-score_honest:.4f}")

# Save metrics
metrics = {
    'rows_total': int(q1['total_rows'][0]),
    'unique_content_ids': int(q1['unique_content_ids'][0]),
    'unique_clients': int(q1['unique_clients'][0]),
    'grain_verified': bool(q1['grain_is_content_id'][0]),
    'pct_available': float(q3['pct_available'][0]),
    'declining_rows': int(q2['declining'][0]),
    'roc_auc_honest': round(float(score_honest), 4),
    'roc_auc_leaked': round(float(score_leaked), 4),
    'leakage_lift': round(float(score_leaked-score_honest), 4)
}
with open(OUT / 'ml04_contract_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\nSaved -> work/outputs/ml04_contract_metrics.json")
print(json.dumps(metrics, indent=2))
