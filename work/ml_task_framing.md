# ML Task Framing: Refresh & Content Opportunity Scoring

- **Provisional Selected Lane:** Core Lane 2: Refresh / Content Opportunity Scoring
- **Date:** July 8, 2026
- **Intern:** Huzaifah (Huzaifah1805)

---

## 1. Mapping the Lane to the Machine Learning Loop

Applying the standard ML Loop framework (`World -> Data -> Features -> Model -> Output -> Decision -> Action -> Changed World`), we map our lane as follows:

```text
               World (User search behaviors, Google algorithm updates)
                                  │
                                  ▼
               Data (90d impressions, clicks, scroll rates, days since update)
                                  │
                                  ▼
               Features (Normalized CTR, position tiers, log impressions, age)
                                  │
                                  ▼
               Model (Random Forest or Decision Tree Classifier)
                                  │
                                  ▼
               Output (Probability of traffic decline + Reason Codes)
                                  │
                                  ▼
               Decision (Prioritize the top-K pages for editorial review)
                                  │
                                  ▼
               Action (Content Refresh: update stats, rewrite title, merge pages)
                                  │
                                  ▼
               Changed World (Restored search visibility, recovered traffic)
```

---

## 2. ML Task Definition

### Task Type: Ranking & Binary Classification
The core technical task is formulated as a **Binary Classification** problem that predicts the probability of a page declining in the future. We then sort this output to solve a **Ranking** problem: surfacing a prioritized queue of the top-K pages that represent the greatest risk or opportunity.

### The Target / Proxy Label
- **Proxy Label (In-Sample / Current Window):** `is_declining_label = (trend_direction == "down")`. This indicates whether the page's organic performance was already in decline during the observation window.
- **Production Target (Future-Looking):** We define a predictive target using separate time windows to prevent data leakage:
  $$\text{Features (computed over prior 90 days)} \rightarrow \text{Target (1 if organic traffic drops } \ge 20\% \text{ in the next 30 days, else 0)}$$

### Success Metrics
Because this is a queue-prioritization system for human review, the primary metric is **Precision@K** (where $K$ matches the editorial team's weekly capacity, e.g., $K=20$ or $K=50$):
$$\text{Precision@K} = \frac{\text{Number of actually declining pages in the top K ranked suggestions}}{K}$$

- **Secondary Metrics:** 
  - **Average Precision (AP) / PR AUC:** To measure the overall ranking quality across the entire list.
  - **ROC AUC:** To measure the model's ability to discriminate between declining and stable pages.

---

## 3. Actionable Output & Business Decisions

The model output is not just a raw probability; it is paired with **reason codes** that map directly to real-world content actions:

| Surfaced Signal | Actionable Reason Code | Targeted Content Action |
|---|---|---|
| High search volume + average position decay (page 1 to page 2 drift) | `page_one_decay_risk` | **SEO Protection:** Immediate content refresh (update outdated statistics, references, and external links) to protect page-one placement. |
| High impressions + low CTR relative to position tier | `low_ctr_visible_page` | **Metadata Optimization:** Rewrite page titles and meta descriptions to improve user search intent matching. |
| Stale update age ($\ge 180$ days) + high impressions | `stale_visible_page` | **Content Expansion:** Audit the content layout, update the publication date, and rewrite thin sections. |
| Low engagement rates / low scroll rates | `low_engagement_visible_page` | **On-Page UX Audit:** Improve internal linking, add relevant images/videos, and move key answers higher up the page. |

---

## 4. Why ML Beats a Fixed Rule

A traditional rule-based system (e.g., "Flag any page that is stale and has over 500 impressions") falls short in three critical ways that machine learning solves:

1.  **Non-Linear Interaction of Signals:**
    A page does not decline due to a single factor. An older page (`content_age_days = 200`) with slightly lower CTR might be perfectly healthy if its position is rising, whereas a younger page with the same CTR could be in severe decay. Fixed linear rules cannot capture these complex, non-linear conditional splits. A Decision Tree/Random Forest automatically learns these branches (e.g., splitting on `avg_position` first, then branching differently for high vs. low `content_age`).
2.  **Context-Aware Thresholds (Noise Filtering):**
    Fixed rules use arbitrary thresholds (like `impressions_90d >= 500`). However, a 10% drop in traffic for a page with 10,000 impressions is a significant signal, while a 50% drop for a page with 50 impressions is likely pure statistical noise. ML models learn to weigh variance and volume dynamically, preventing low-volume pages from polluting the review queue.
3.  **Measurable Performance Lift:**
    Our local baseline run demonstrated this gap empirically:
    - **Hand-written rule baseline:** Precision@50 = **0.240** (only 12 of the top 50 suggested pages were actually declining).
    - **Random Forest model:** Precision@50 = **0.740** (37 of the top 50 suggested pages were actually declining).
    
    The ML model delivers a **~3x precision lift**, meaning editors waste 76% less time reviewing healthy pages and focus almost entirely on actual opportunities.
