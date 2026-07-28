"""
ML-10: Content Action Playbook — execution script.
Generates ranked queue, reason codes, archetype mapping, figures, and metrics JSON.
Outputs to work/outputs/ and work/figures/.
"""
import sys, json, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

warnings.filterwarnings('ignore')

ROOT   = Path(__file__).resolve().parents[2]
FEAT   = ROOT / 'data/processed/refresh_feature_vector.csv'
PRED   = ROOT / 'data/processed/model_predictions.csv'
OUT    = ROOT / 'work/outputs'
FIGS   = ROOT / 'work/figures'
OUT.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ── 1. Load and merge ──────────────────────────────────────────────────────────
features = pd.read_csv(FEAT)
preds    = pd.read_csv(PRED)

# Re-run Gradient Boosting for clean scores across full dataset
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

FEATURE_COLS = [
    'log_impressions_90d', 'log_clicks_90d', 'log_sessions_90d',
    'log_ai_sessions_90d', 'ctr', 'avg_position', 'days_since_last_update',
    'content_age_days', 'search_volume', 'engagement_rate', 'scroll_rate',
    'impressions_last_30d', 'impressions_prev_30d', 'has_clicks',
    'has_ai_sessions', 'measurable_opportunity', 'competition', 'cpc'
]
LABEL_COL = 'is_declining_label'

df = features.dropna(subset=FEATURE_COLS + [LABEL_COL]).copy()
X  = df[FEATURE_COLS]
y  = df[LABEL_COL]

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                       learning_rate=0.05, subsample=0.8,
                                       random_state=42))
])
pipe.fit(X, y)
df['decline_score'] = pipe.predict_proba(X)[:, 1]
print(f"Model fitted. Rows scored: {len(df):,}")

# ── 2. Reason codes ───────────────────────────────────────────────────────────
def assign_reason_codes(row):
    codes = []
    if row['days_since_last_update'] > 180:
        codes.append('STALE_CONTENT')
    if row['avg_position'] <= 10 and row['ctr'] < 0.02:
        codes.append('PAGE1_LOW_CTR')
    if row['impressions_prev_30d'] > 0:
        mom = row['impressions_last_30d'] / (row['impressions_prev_30d'] + 1)
        if mom < 0.7:
            codes.append('IMPRESSION_DROP')
    if row['engagement_rate'] < 0.05:
        codes.append('LOW_ENGAGEMENT')
    if row['search_volume'] > 100 and row['decline_score'] > 0.6:
        codes.append('HIGH_VOLUME_AT_RISK')
    if row['content_age_days'] < 60:
        codes.append('NEW_CONTENT')
    return '|'.join(codes) if codes else 'GENERAL_DECAY'

df['reason_codes'] = df.apply(assign_reason_codes, axis=1)

# ── 3. Archetype → Action mapping ────────────────────────────────────────────
def assign_action(row):
    codes = row['reason_codes'].split('|')
    score = row['decline_score']
    age   = row['content_age_days']

    if 'NEW_CONTENT' in codes:
        return 'MONITOR'          # too early to refresh

    if 'HIGH_VOLUME_AT_RISK' in codes and score > 0.75:
        return 'FULL_REWRITE'     # high-value declining: major investment justified

    if 'PAGE1_LOW_CTR' in codes:
        return 'META_TITLE_REFRESH'  # quick win: fix title/description

    if 'STALE_CONTENT' in codes and 'IMPRESSION_DROP' in codes:
        return 'CONTENT_UPDATE'   # stale + dropping: targeted refresh

    if 'IMPRESSION_DROP' in codes and score > 0.6:
        return 'CONTENT_UPDATE'

    if 'LOW_ENGAGEMENT' in codes:
        return 'UX_IMPROVEMENT'   # good traffic, bad engagement: structure/links

    if score > 0.65:
        return 'REVIEW_SCHEDULE'  # declining but no clear signal: human review

    return 'MONITOR'

df['recommended_action'] = df.apply(assign_action, axis=1)

# ── 4. Priority tier ─────────────────────────────────────────────────────────
def assign_priority(row):
    if row['recommended_action'] in ('FULL_REWRITE',) and row['decline_score'] > 0.75:
        return 'P1_URGENT'
    if row['recommended_action'] in ('CONTENT_UPDATE', 'META_TITLE_REFRESH') and row['decline_score'] > 0.6:
        return 'P2_HIGH'
    if row['recommended_action'] in ('UX_IMPROVEMENT', 'REVIEW_SCHEDULE'):
        return 'P3_MEDIUM'
    return 'P4_WATCH'

df['priority']         = df.apply(assign_priority, axis=1)
df['refresh_urgency']  = df['decline_score'].round(4)

# ── 5. Ranked queue export ────────────────────────────────────────────────────
QUEUE_COLS = [
    'content_id', 'client_id', 'decline_score', 'refresh_urgency',
    'priority', 'recommended_action', 'reason_codes',
    'days_since_last_update', 'avg_position', 'ctr',
    'impressions_90d', 'search_volume', 'engagement_rate',
    'content_age_days', 'is_declining_label'
]
queue = df[QUEUE_COLS].sort_values('decline_score', ascending=False).reset_index(drop=True)
queue['rank'] = queue.index + 1

queue_path = OUT / 'ml10_ranked_queue.csv'
queue.to_csv(queue_path, index=False)
print(f"Queue exported: {len(queue):,} rows -> {queue_path}")

# Top-50 for human review
top50 = queue.head(50)
print(f"\nTop-50 action distribution:")
print(top50['recommended_action'].value_counts().to_string())
print(f"\nTop-50 priority distribution:")
print(top50['priority'].value_counts().to_string())

# ── 6. Figures ────────────────────────────────────────────────────────────────
PALETTE = {
    'FULL_REWRITE':        '#ef4444',
    'CONTENT_UPDATE':      '#f97316',
    'META_TITLE_REFRESH':  '#eab308',
    'UX_IMPROVEMENT':      '#3b82f6',
    'REVIEW_SCHEDULE':     '#8b5cf6',
    'MONITOR':             '#6b7280',
}

# Figure 1: Action mix across full queue
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor='#0d1117')
for ax in axes:
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#e6edf3')
    ax.spines['bottom'].set_color('#30363d')
    ax.spines['left'].set_color('#30363d')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Pie: full queue action distribution
action_counts = queue['recommended_action'].value_counts()
colors = [PALETTE.get(a, '#6b7280') for a in action_counts.index]
axes[0].pie(action_counts.values, labels=action_counts.index, colors=colors,
            autopct='%1.1f%%', startangle=140,
            textprops={'color': '#e6edf3', 'fontsize': 9})
axes[0].set_title('Action Mix — Full Queue', color='#e6edf3', fontsize=12, pad=15)

# Bar: score distribution by action (top 500)
top500 = queue.head(500)
action_order = ['FULL_REWRITE','CONTENT_UPDATE','META_TITLE_REFRESH',
                'UX_IMPROVEMENT','REVIEW_SCHEDULE','MONITOR']
means = top500.groupby('recommended_action')['decline_score'].mean().reindex(action_order).dropna()
bars  = axes[1].bar(range(len(means)), means.values,
                    color=[PALETTE.get(a, '#6b7280') for a in means.index],
                    edgecolor='none', width=0.6)
axes[1].set_xticks(range(len(means)))
axes[1].set_xticklabels([a.replace('_', '\n') for a in means.index], fontsize=8, color='#e6edf3')
axes[1].set_ylabel('Mean Decline Score', color='#8b949e')
axes[1].set_title('Mean Score by Action (Top 500)', color='#e6edf3', fontsize=12)
axes[1].set_ylim(0, 1)
axes[1].yaxis.label.set_color('#8b949e')
for bar, val in zip(bars, means.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', color='#e6edf3', fontsize=9)

plt.tight_layout()
fig1_path = FIGS / 'ml10_action_mix.png'
plt.savefig(fig1_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"Figure 1 saved: {fig1_path}")

# Figure 2: Score distribution + threshold line
fig2, ax2 = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
ax2.set_facecolor('#161b22')
ax2.tick_params(colors='#e6edf3')
for spine in ax2.spines.values():
    spine.set_color('#30363d')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax2.hist(queue[queue['is_declining_label']==0]['decline_score'], bins=50,
         color='#3b82f6', alpha=0.6, label='Not Declining', edgecolor='none')
ax2.hist(queue[queue['is_declining_label']==1]['decline_score'], bins=50,
         color='#ef4444', alpha=0.6, label='Declining', edgecolor='none')
ax2.axvline(0.6, color='#f59e0b', linestyle='--', linewidth=1.5, label='Action Threshold (0.6)')
ax2.set_xlabel('Decline Probability Score', color='#8b949e')
ax2.set_ylabel('Page Count', color='#8b949e')
ax2.set_title('Score Distribution by True Label', color='#e6edf3', fontsize=13)
legend = ax2.legend(framealpha=0, labelcolor='#e6edf3', fontsize=10)
plt.tight_layout()
fig2_path = FIGS / 'ml10_score_distribution.png'
plt.savefig(fig2_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"Figure 2 saved: {fig2_path}")

# Figure 3: Top reason codes
reason_flat = []
for codes in queue.head(500)['reason_codes']:
    reason_flat.extend(codes.split('|'))
reason_series = pd.Series(reason_flat).value_counts().head(8)

fig3, ax3 = plt.subplots(figsize=(10, 5), facecolor='#0d1117')
ax3.set_facecolor('#161b22')
ax3.tick_params(colors='#e6edf3')
for spine in ax3.spines.values(): spine.set_color('#30363d')
ax3.spines['top'].set_visible(False); ax3.spines['right'].set_visible(False)

colors3 = ['#8b5cf6'] * len(reason_series)
ax3.barh(range(len(reason_series)), reason_series.values[::-1],
         color=colors3, edgecolor='none')
ax3.set_yticks(range(len(reason_series)))
ax3.set_yticklabels(reason_series.index[::-1], color='#e6edf3', fontsize=10)
ax3.set_xlabel('Count in Top-500', color='#8b949e')
ax3.set_title('Top Reason Codes (Top-500 queue)', color='#e6edf3', fontsize=13)
plt.tight_layout()
fig3_path = FIGS / 'ml10_top_reason_codes.png'
plt.savefig(fig3_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.close()
print(f"Figure 3 saved: {fig3_path}")

# ── 7. Metrics JSON ───────────────────────────────────────────────────────────
p1_count = len(queue[queue['priority'] == 'P1_URGENT'])
p2_count = len(queue[queue['priority'] == 'P2_HIGH'])
action_mix = queue['recommended_action'].value_counts().to_dict()
top50_precision = top50['is_declining_label'].mean()

metrics = {
    'total_pages_scored':   len(queue),
    'action_threshold':     0.6,
    'p1_urgent_count':      int(p1_count),
    'p2_high_count':        int(p2_count),
    'top50_precision':      round(float(top50_precision), 4),
    'action_mix_full_queue': {k: int(v) for k, v in action_mix.items()},
    'mean_score_declining': round(float(queue[queue['is_declining_label']==1]['decline_score'].mean()), 4),
    'mean_score_not_declining': round(float(queue[queue['is_declining_label']==0]['decline_score'].mean()), 4),
    'model_used':           'Gradient Boosting (n_estimators=200, max_depth=4)',
    'validation_p50':       0.772,
    'figures': ['ml10_action_mix.png', 'ml10_score_distribution.png', 'ml10_top_reason_codes.png'],
}
metrics_path = OUT / 'ml10_playbook_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved: {metrics_path}")
print(json.dumps(metrics, indent=2))
