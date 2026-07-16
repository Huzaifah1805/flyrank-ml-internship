"""
ML-08 Capstone Modeling Lane - execution script
Mirrors w05_model.ipynb logic exactly.
Outputs: work/outputs/ml08_model_metrics.json
"""
import os
import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / 'data/processed/refresh_feature_vector.csv'
BASELINE_PATH = ROOT / 'data/processed/baseline_refresh_queue.csv'
OUT_DIR = ROOT / 'work/outputs'
OUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def precision_at_k(y_true, y_scores, k):
    df = pd.DataFrame({'y': list(y_true), 'score': list(y_scores)})
    top = df.sort_values('score', ascending=False).head(min(k, len(df)))
    return float(top['y'].mean())

print("Loading data...")
df = pd.read_csv(DATA_PATH)
baseline_df = pd.read_csv(BASELINE_PATH)
df = df.merge(baseline_df[['content_id','baseline_refresh_score']], on='content_id', how='left')

# Feature engineering
df['ctr_to_pos'] = df['ctr'] / (df['avg_position'] + 1.0)
df['stale_age_interaction'] = df['days_since_last_update'] * df['content_age_days']
df['session_volume_efficiency'] = df['sessions_90d'] / (df['search_volume'] + 1.0)
df['engagement_intensity'] = df['engaged_sessions_90d'] / (df['sessions_90d'] + 1.0)
df['ai_traffic_ratio'] = df['ai_sessions_90d'] / (df['sessions_90d'] + 1.0)

numeric_features = [
    'search_volume','competition','cpc','word_count','char_count',
    'log_impressions_90d','log_clicks_90d','log_sessions_90d','log_ai_sessions_90d',
    'days_with_impressions','days_with_sessions','content_age_days',
    'days_since_last_update','ctr','avg_position','engagement_rate',
    'scroll_rate','ai_traffic_pct','ctr_to_pos','stale_age_interaction',
    'session_volume_efficiency','engagement_intensity','ai_traffic_ratio'
]
categorical_features = [
    'competition_level','content_type','main_intent','age_tier',
    'freshness_tier','word_count_tier','impression_tier','position_tier'
]

cat_df = pd.get_dummies(df[categorical_features].fillna('unknown'), dtype=float)
num_df = df[numeric_features].fillna(0).replace([np.inf, -np.inf], 0)
X = pd.concat([num_df, cat_df], axis=1)
y = (df['trend_direction'] == 'down').astype(int)
groups = df['client_id'].astype(str)

print(f"Rows: {len(df):,} | Features: {X.shape[1]} | Positive rate: {y.mean():.3f}")
print(f"Unique clients: {groups.nunique()}")

models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE))
    ]),
    'Decision Tree (D5)': DecisionTreeClassifier(
        class_weight='balanced', max_depth=5, min_samples_leaf=50, random_state=RANDOM_STATE
    ),
    'Random Forest': RandomForestClassifier(
        class_weight='balanced_subsample', max_depth=10, min_samples_leaf=25,
        n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        max_depth=4, min_samples_leaf=20, n_estimators=100, learning_rate=0.05,
        random_state=RANDOM_STATE
    )
}

results = {name: {'p50': [], 'roc_auc': [], 'avg_precision': []} for name in list(models.keys()) + ['ML-07 Baseline']}
gkf = GroupKFold(n_splits=5)
fitted_models = {}
last_val_idx = None
last_scores = {}

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    baseline_val = df.iloc[val_idx]['baseline_refresh_score'].fillna(0).values

    results['ML-07 Baseline']['p50'].append(precision_at_k(y_val, baseline_val, 50))
    results['ML-07 Baseline']['roc_auc'].append(roc_auc_score(y_val, baseline_val))
    results['ML-07 Baseline']['avg_precision'].append(average_precision_score(y_val, baseline_val))

    for name, model in models.items():
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_val)[:, 1]
        results[name]['p50'].append(precision_at_k(y_val, scores, 50))
        results[name]['roc_auc'].append(roc_auc_score(y_val, scores))
        results[name]['avg_precision'].append(average_precision_score(y_val, scores))
        if fold == 4:
            fitted_models[name] = model
            last_scores[name] = scores
    if fold == 4:
        last_val_idx = val_idx
    print(f"Fold {fold+1}/5 done")

print("\n=== MODEL vs BASELINE COMPARISON ===")
rows = []
for name, m in results.items():
    rows.append({
        'Model': name,
        'Precision@50': round(np.mean(m['p50']), 4),
        'ROC-AUC': round(np.mean(m['roc_auc']), 4),
        'Avg Precision': round(np.mean(m['avg_precision']), 4)
    })
compare_df = pd.DataFrame(rows).sort_values('Precision@50', ascending=False)
print(compare_df.to_string(index=False))

# Permutation importance on best model (Gradient Boosting)
print("\n=== PERMUTATION IMPORTANCE (Gradient Boosting, last fold) ===")
best_model = fitted_models['Gradient Boosting']
X_val_last = X.iloc[last_val_idx]
y_val_last = y.iloc[last_val_idx]

perm = permutation_importance(
    best_model, X_val_last, y_val_last,
    n_repeats=5, random_state=RANDOM_STATE, scoring='roc_auc'
)
feat_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': perm.importances_mean
}).sort_values('importance', ascending=False).head(15)
print(feat_imp.to_string(index=False))

# Error analysis
print("\n=== ERROR ANALYSIS (top-50, last fold) ===")
val_df_err = df.iloc[last_val_idx].copy()
val_df_err['model_score'] = last_scores['Gradient Boosting']
val_df_err['actual_decline'] = y_val_last.values
top50 = val_df_err.sort_values('model_score', ascending=False).head(50)
fp = top50[top50['actual_decline'] == 0]
print(f"True positives in top-50: {(top50.actual_decline==1).sum()}")
print(f"False positives in top-50: {len(fp)}")
print("\nFalse Positive sample:")
print(fp[['content_id','days_since_last_update','ctr','avg_position','trend_direction']].head(5).to_string(index=False))

# Save metrics
ml08_metrics = {
    'models': {name: {
        'precision_at_50': round(float(np.mean(m['p50'])), 4),
        'roc_auc': round(float(np.mean(m['roc_auc'])), 4),
        'avg_precision': round(float(np.mean(m['avg_precision'])), 4)
    } for name, m in results.items()},
    'baseline_p50': 0.500,
    'best_model': 'Gradient Boosting',
    'validation': '5-fold GroupKFold by client_id'
}
with open(OUT_DIR / 'ml08_model_metrics.json', 'w') as f:
    json.dump(ml08_metrics, f, indent=2)
print("\nSaved -> work/outputs/ml08_model_metrics.json")
print(json.dumps(ml08_metrics, indent=2))
