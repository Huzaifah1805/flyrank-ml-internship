# Capstone Report — Refresh / Content Opportunity Scoring

- **Author:** Huzaifah (Huzaifah1805)
- **Lane:** Core Lane 2: Refresh / Content Opportunity Scoring
- **Repo:** [Huzaifah1805/flyrank-ml-internship](https://github.com/Huzaifah1805/flyrank-ml-internship)
- **Date:** July 8, 2026

## 1. Problem framing
This capstone supports the **content refresh prioritization decision**. 
- **Unit of analysis:** A single pseudonymized content item (page).
- **Output:** A ranked queue of review candidates with risk probabilities.
- **Action:** A human editor reviews the top-K flagged pages to update stale facts, fix intent mismatch, or expand thin sections.
- **Cost of a wrong call:** False positives waste expensive editorial capacity. False negatives result in a permanent loss of search discoverability.
- **Why ML helps:** Search decay is non-linear. A page with 10k impressions dropping 5% is a crisis; a page with 100 impressions dropping 50% is statistical noise. A fixed rule cannot weigh these conditional boundaries efficiently across thousands of pages, but a machine learning model can.

## 2. Data safety
- **Data used:** I used the pseudonymized starter slice (`refresh_feature_vector.csv`), maintaining strict isolation of the 90-day feature window from the outcome.
- **Excluded columns (Leakage Prevention):** I deliberately excluded `impressions_last_30d`, `impressions_prev_30d`, `trend_pct`, and `trend_direction` from the model's feature set. These are target-window measurements that directly define the label; including them would cause severe data leakage and artificially inflate the model's accuracy.
- **Pseudonymous IDs:** `client_id` and `content_id` were used exclusively for grouping and validation splits, never as features.
- No client-identifying data exists anywhere in my `work/` folder.

## 3. Baseline
My baseline was a transparent, human-readable composite rule (`baseline_refresh_score`) built from visibility, freshness, and position opportunity. 
- It serves as a fair comparison because it represents how a smart SEO manager would build a SQL query to find declining pages today.
- **Baseline Precision@50:** **0.464** (it correctly flagged 23 actually declining pages in its top 50 recommendations).

## 4. Model / analysis
- **Method:** I engineered safe interaction features (e.g., `session_volume_efficiency`, `stale_age_interaction`) and trained Logistic Regression, Random Forest, and Gradient Boosting Classifiers.
- **Why it fits:** Gradient Boosting and Random Forests excel at capturing the non-linear boundaries between safe variance and true decay.
- **Proxy Definition:** A page is marked as declining (`1`) if `trend_direction == 'down'` (which equates to an impression drop >20% month-over-month).

## 5. Evaluation
- **Split Strategy:** I implemented a **5-fold GroupKFold validation split grouped by `client_id`**. This is critical: if we simply shuffled rows randomly, the model might memorize a specific client's site structure and overfit. Grouping by client proves the model generalizes to entirely new websites.
- **Metrics (Average over 5 folds):**
  - Base Rate: 0.542
  - **Baseline Rule Precision@50:** 0.464
  - **Random Forest Precision@50:** 0.720
  - **Gradient Boosting Precision@50:** **0.824**
- **Error Analysis:** The Gradient Boosting model achieved an 82.4% precision in its top 50 recommendations, nearly doubling the efficiency of the fixed baseline rule. False positives typically occurred on pages with very low overall search volume, where percentage drops are erratic due to noise.

## 6. Interpretation
- **What the model found:** Feature importance analysis revealed that `ctr`, `avg_position`, and the engineered `session_volume_efficiency` were the dominant drivers of the prediction. 
- **Non-linear effects:** The tree-based models successfully learned that `content_age_days` only becomes a risk factor when combined with a dropping `ctr_to_pos` ratio, something the linear Baseline rule failed to capture.

## 7. Recommendation
- **Ranked Actions:** The pipeline outputs `model_predictions.csv`, sorted by `prob_Gradient_Boosting`. A FlyRank editor should start at the top of this list and review the top 50 pages weekly.
- **Limits and Cautious Claims:** These predictions are observational and directional. Surfacing a page implies a high probability of search demand decay; it does *not* guarantee that an editorial refresh will reverse the trend (which would require a causal experiment to prove). 

## 8. Reproducibility
To reproduce this entire pipeline from a fresh clone, run the following commands in the terminal:
```powershell
# 1. Activate the environment
venv\Scripts\activate

# 2. Re-prepare the base feature vector
python scripts\01_prepare_features.py

# 3. Calculate baseline scores
python scripts\02_baseline_score.py

# 4. Run the full Capstone ML pipeline (trains models, generates metrics and figures)
$env:PYTHONIOENCODING="utf-8"; python work\scripts\train_capstone.py
```
- **Random Seeds:** Fixed at `RANDOM_STATE = 42` in `train_capstone.py` and `GroupKFold`.
- **Environment:** Relies on `scikit-learn`, `pandas`, `numpy`, and `matplotlib` as installed via `requirements.txt`.
