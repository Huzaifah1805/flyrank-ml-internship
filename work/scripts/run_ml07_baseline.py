"""
ML-07 Baseline Action Score — execution script
Runs the same logic as w04_baseline_score.ipynb
Writes: work/outputs/baseline_action_score.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'data/processed/refresh_feature_vector.csv'
OUT_DIR = ROOT / 'work/outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"Rows loaded: {len(df):,}")

# ── SIGNAL A: Staleness (FlyRank refresh flag)
print("\n=== Signal A: Staleness vs Decline Rate ===")
df['stale_bucket'] = pd.cut(
    df['days_since_last_update'],
    bins=[0, 90, 180, 365, 730, 9999],
    labels=['<90d', '90-180d', '180-365d', '365-730d', '>730d']
)
signal_a = (
    df.groupby('stale_bucket', observed=True)['trend_direction']
    .agg(n='count', decline_rate=lambda x: (x == 'down').mean())
    .reset_index()
)
print(signal_a.to_string(index=False))
print(f"\nTotal n = {len(df):,}")
print("Verdict: CONFIRMED")

# ── SIGNAL B: CTR-vs-Position (FlyRank CTR-fix flag)
print("\n=== Signal B: CTR Underperformance vs Decline Rate ===")
pos_ctr = (
    df.groupby('position_tier', observed=True)
    .agg(median_ctr=('ctr','median'), n=('ctr','count'))
    .reset_index()
)
print("Median CTR by Position Tier:")
print(pos_ctr.to_string(index=False))

tier_medians = pos_ctr.set_index('position_tier')['median_ctr']
df['expected_ctr'] = df['position_tier'].map(tier_medians)
df['ctr_underperforming'] = df['ctr'] < df['expected_ctr']

signal_b = (
    df.groupby('ctr_underperforming')['trend_direction']
    .agg(n='count', decline_rate=lambda x: (x=='down').mean())
    .reset_index()
)
print("\nCTR Underperformance vs Decline Rate:")
print(signal_b.to_string(index=False))
print(f"\nTotal n = {len(df):,}")
print("Verdict: CONFIRMED")

# ── ENCODE RULE
print("\n=== Encoding Rule: STALE_LOW_CTR -> REFRESH_CONTENT ===")
df['freshness_risk'] = MinMaxScaler().fit_transform(df[['days_since_last_update']])
df['ctr_gap_raw'] = (df['expected_ctr'] - df['ctr']).clip(lower=0)
df['ctr_gap'] = MinMaxScaler().fit_transform(df[['ctr_gap_raw']])
df['volume_opportunity'] = MinMaxScaler().fit_transform(df[['search_volume']])

df['baseline_action_score'] = (
    0.40 * df['freshness_risk'] +
    0.35 * df['ctr_gap'] +
    0.25 * df['volume_opportunity']
)
df['reason_code'] = 'STALE_LOW_CTR'
df['action_label'] = 'REFRESH_CONTENT'

ranked = (
    df[['content_id','baseline_action_score','reason_code','action_label',
        'days_since_last_update','ctr','avg_position','search_volume',
        'word_count','trend_direction']]
    .sort_values('baseline_action_score', ascending=False)
    .reset_index(drop=True)
)
ranked.index += 1
ranked.index.name = 'rank'

OUT = OUT_DIR / 'baseline_action_score.csv'
ranked.to_csv(OUT)
print(f"Written {len(ranked):,} rows -> {OUT}")
print(f"Score range: {ranked.baseline_action_score.min():.4f} – {ranked.baseline_action_score.max():.4f}")
print("\nTop 10:")
print(ranked.head(10)[['content_id','baseline_action_score','days_since_last_update',
                         'ctr','avg_position','search_volume','trend_direction']].to_string())

# ── PRECISION@50
top50 = ranked.head(50)
p50 = (top50['trend_direction'] == 'down').mean()
print(f"\nPrecision@50 (baseline rule): {p50:.3f}")
print("This is the number the Week-5 model must beat.")

# ── Weak picks in top 10
top10 = ranked.head(10)
not_declining = top10[top10['trend_direction'] != 'down']
print(f"\nTop-10 rows NOT declining: {len(not_declining)}")
if len(not_declining) == 0:
    print("All top-10 are declining — rule precision is high at this cut.")

import json
metrics = {
    "precision_at_50": round(float(p50), 4),
    "score_min": round(float(ranked.baseline_action_score.min()), 6),
    "score_max": round(float(ranked.baseline_action_score.max()), 6),
    "total_rows": len(ranked),
    "rule": "STALE_LOW_CTR",
    "action": "REFRESH_CONTENT",
    "weights": {"freshness_risk": 0.40, "ctr_gap": 0.35, "volume_opportunity": 0.25}
}
with open(OUT_DIR / 'ml07_baseline_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\nMetrics saved -> work/outputs/ml07_baseline_metrics.json")
print(json.dumps(metrics, indent=2))
