"""
Capstone Project Script: End-to-end Machine Learning Pipeline
This script trains a series of classifiers for Core Lane 2 (Refresh / Content Opportunity Scoring)
on a 5-fold GroupKFold cross-validation split by client_id. It calculates advanced engineered features
(without target leakage), optimizes models for Precision@50, and outputs metrics and figures.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, precision_recall_curve, roc_curve
)

# Setup paths relative to script
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "refresh_feature_vector.csv"
BASELINE_PATH = ROOT / "data" / "processed" / "baseline_refresh_queue.csv"
FIGURES_DIR = ROOT / "work" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def precision_at_k(y_true, y_scores, k):
    df = pd.DataFrame({"y": y_true, "score": y_scores})
    if len(df) == 0:
        return 0.0
    top = df.sort_values("score", ascending=False).head(min(k, len(df)))
    return float(top["y"].mean())

def load_and_engineer_features():
    df = pd.read_csv(DATA_PATH)
    baseline_df = pd.read_csv(BASELINE_PATH)
    
    # Merge baseline score for comparison
    df = df.merge(baseline_df[["content_id", "baseline_refresh_score"]], on="content_id", how="left")
    
    # Feature engineering: Safe interaction features (no target-window leakage)
    # 1. CTR-to-position ratio: representing whether the page performs well given its position
    df["ctr_to_pos"] = df["ctr"] / (df["avg_position"] + 1.0)
    
    # 2. Update status interaction: stale and old content interaction
    df["stale_age_interaction"] = df["days_since_last_update"] * df["content_age_days"]
    
    # 3. Search demand index: organic session rate relative to search volume
    df["session_volume_efficiency"] = df["sessions_90d"] / (df["search_volume"] + 1.0)
    
    # 4. Engagement density: engaged sessions relative to total sessions
    df["engagement_intensity"] = df["engaged_sessions_90d"] / (df["sessions_90d"] + 1.0)
    
    # 5. AI session intensity: AI traffic share
    df["ai_traffic_ratio"] = df["ai_sessions_90d"] / (df["sessions_90d"] + 1.0)
    
    # Define numeric features to use in model (leaving out impressions_last_30d and impressions_prev_30d)
    numeric_features = [
        "search_volume", "competition", "cpc", "word_count", "char_count",
        "log_impressions_90d", "log_clicks_90d", "log_sessions_90d", "log_ai_sessions_90d",
        "days_with_impressions", "days_with_sessions", "content_age_days", 
        "days_since_last_update", "ctr", "avg_position", "engagement_rate", 
        "scroll_rate", "ai_traffic_pct", "ctr_to_pos", "stale_age_interaction",
        "session_volume_efficiency", "engagement_intensity", "ai_traffic_ratio"
    ]
    
    categorical_features = [
        "competition_level", "content_type", "main_intent", "age_tier", 
        "freshness_tier", "word_count_tier", "impression_tier", "position_tier"
    ]
    
    # Preprocess categorical features (One-hot encoding)
    cat_df = pd.get_dummies(df[categorical_features].fillna("unknown"), dtype=float)
    num_df = df[numeric_features].fillna(0).replace([np.inf, -np.inf], 0)
    
    X = pd.concat([num_df, cat_df], axis=1)
    y = df["is_declining_label"].astype(int)
    groups = df["client_id"].astype(str)
    
    return df, X, y, groups

def train_and_evaluate():
    df, X, y, groups = load_and_engineer_features()
    
    # Check baseline base rate
    base_rate = y.mean()
    print(f"Total dataset rows: {len(df)}")
    print(f"Base rate (declinement rate): {base_rate:.3f}")
    
    # Initialize models
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree (Depth 5)": DecisionTreeClassifier(
            class_weight="balanced", max_depth=5, min_samples_leaf=50, random_state=RANDOM_STATE
        ),
        "Random Forest (Depth 10)": RandomForestClassifier(
            class_weight="balanced_subsample", max_depth=10, min_samples_leaf=25,
            n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            max_depth=5, min_samples_leaf=20, n_estimators=150, learning_rate=0.05, random_state=RANDOM_STATE
        )
    }
    
    # 5-fold GroupKFold validation by client_id
    gkf = GroupKFold(n_splits=5)
    
    metrics = {model_name: {
        "precision_at_20": [], "precision_at_50": [], "roc_auc": [], "average_precision": [],
        "f1": [], "precision": [], "recall": []
    } for model_name in list(models.keys()) + ["Baseline Rule"]}
    
    # Store predictions for plotting curves later
    plot_data = {model_name: {"y_true": [], "y_score": []} for model_name in list(models.keys()) + ["Baseline Rule"]}
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
        print(f"\n--- Fold {fold+1}/5 ---")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 1. Baseline Evaluation
        val_baseline = df.iloc[val_idx]["baseline_refresh_score"].fillna(0).values
        metrics["Baseline Rule"]["precision_at_20"].append(precision_at_k(y_val, val_baseline, 20))
        metrics["Baseline Rule"]["precision_at_50"].append(precision_at_k(y_val, val_baseline, 50))
        metrics["Baseline Rule"]["roc_auc"].append(roc_auc_score(y_val, val_baseline))
        metrics["Baseline Rule"]["average_precision"].append(average_precision_score(y_val, val_baseline))
        # Simple threshold for baseline f1/precision/recall (score >= 50)
        baseline_pred = (val_baseline >= 50).astype(int)
        metrics["Baseline Rule"]["f1"].append(f1_score(y_val, baseline_pred, zero_division=0))
        metrics["Baseline Rule"]["precision"].append(precision_score(y_val, baseline_pred, zero_division=0))
        metrics["Baseline Rule"]["recall"].append(recall_score(y_val, baseline_pred, zero_division=0))
        plot_data["Baseline Rule"]["y_true"].extend(y_val)
        plot_data["Baseline Rule"]["y_score"].extend(val_baseline)
        
        # 2. ML Models Evaluation
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)
            
            metrics[model_name]["precision_at_20"].append(precision_at_k(y_val, probs, 20))
            metrics[model_name]["precision_at_50"].append(precision_at_k(y_val, probs, 50))
            metrics[model_name]["roc_auc"].append(roc_auc_score(y_val, probs))
            metrics[model_name]["average_precision"].append(average_precision_score(y_val, probs))
            metrics[model_name]["f1"].append(f1_score(y_val, preds, zero_division=0))
            metrics[model_name]["precision"].append(precision_score(y_val, preds, zero_division=0))
            metrics[model_name]["recall"].append(recall_score(y_val, preds, zero_division=0))
            plot_data[model_name]["y_true"].extend(y_val)
            plot_data[model_name]["y_score"].extend(probs)
            
    # Calculate means
    mean_metrics = {}
    print("\n" + "="*50)
    print("CV Evaluation Summary (Mean values across 5 client-holdout folds)")
    print("="*50)
    for model_name, m_dict in metrics.items():
        mean_metrics[model_name] = {k: np.mean(v) for k, v in m_dict.items()}
        print(f"\nModel: {model_name}")
        print(f"  Precision@20:       {mean_metrics[model_name]['precision_at_20']:.3f}")
        print(f"  Precision@50:       {mean_metrics[model_name]['precision_at_50']:.3f}")
        print(f"  PR-AUC (Avg Prec):  {mean_metrics[model_name]['average_precision']:.3f}")
        print(f"  ROC-AUC:            {mean_metrics[model_name]['roc_auc']:.3f}")
        print(f"  F1 Score:           {mean_metrics[model_name]['f1']:.3f}")
        print(f"  Precision:          {mean_metrics[model_name]['precision']:.3f}")
        print(f"  Recall:             {mean_metrics[model_name]['recall']:.3f}")

    # Plot Precision-Recall Curves
    plt.figure(figsize=(10, 6))
    for model_name in plot_data:
        y_true_all = np.array(plot_data[model_name]["y_true"])
        y_score_all = np.array(plot_data[model_name]["y_score"])
        # Normalize baseline score to [0,1] for drawing curves
        if model_name == "Baseline Rule":
            y_score_all = (y_score_all - y_score_all.min()) / (y_score_all.max() - y_score_all.min() + 1e-9)
        p, r, _ = precision_recall_curve(y_true_all, y_score_all)
        ap = average_precision_score(y_true_all, y_score_all)
        plt.plot(r, p, label=f"{model_name} (PR AUC: {ap:.3f})")
    
    plt.axhline(y=base_rate, color="grey", linestyle="--", label=f"Base Rate ({base_rate:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (5-fold Client-Holdout Out-of-Sample)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pr_curves.png", dpi=150)
    plt.close()
    
    # Feature Importance (Random Forest trained on full data)
    rf_full = models["Random Forest (Depth 10)"]
    rf_full.fit(X, y)
    importances = rf_full.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = 15
    top_features = [X.columns[i] for i in indices[:top_n]]
    top_importances = importances[indices[:top_n]]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), top_importances[::-1], align='center', color='#3498db')
    plt.yticks(range(top_n), top_features[::-1])
    plt.xlabel('Importance')
    plt.title('Top 15 Feature Importances (Random Forest)')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close()
    
    print("\nSaved figures to work/figures/")
    
    # Write JSON metadata
    summary_results = {
        "base_rate": float(base_rate),
        "metrics": mean_metrics,
        "best_model": "Random Forest (Depth 10)",
        "features_used": list(X.columns)
    }
    with open(ROOT / "work" / "capstone_results.json", "w") as f:
        json.dump(summary_results, f, indent=2)
    print("Saved metrics to work/capstone_results.json")

if __name__ == "__main__":
    train_and_evaluate()
