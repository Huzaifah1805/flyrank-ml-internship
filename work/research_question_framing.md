# Research Question Framing: Refresh & Content Opportunity Scoring

- **Provisional Selected Lane:** Core Lane 2: Refresh / Content Opportunity Scoring
- **Date:** July 8, 2026
- **Intern:** Huzaifah (Huzaifah1805)

---

## 1. Core Framework & Definitions

### The Search/Discoverability Question
> *"Which high-visibility content pages are exhibiting patterns of persistent search-performance decline (visibility, impressions, or position decay) and should be prioritized for editor review first?"*

We want to find pages that have sufficient historical search exposure (are "worth fixing") but are showing evidence of structural traffic decline, rather than temporary seasonal fluctuation or random keyword noise.

### Unit of Analysis
The unit of analysis is a **single pseudonymized content item (page)** associated with a client.
- In the dataset: represented by a unique `content_hash_id` (joined with client context via `client_hash_id`).

### The Output of the System
A **ranked priority queue** of pages. Each candidate in the queue will have:
1. A **priority score** (indicating relative urgency for review).
2. **Actionable reason codes** (e.g., `stale_visible_page`, `declining_with_demand`, `page_one_decay_risk`) explaining the underlying signals to help editors.
3. A **confidence tier** (e.g., High, Medium, Low) based on historical volume thresholds.

### The Actionable Decision
A content editor or SEO manager will take one of the following concrete actions:
- **Refresh/Expand:** Update outdated facts/dates, add new sections to address current user intent, or enrich thin sections.
- **Merge/Consolidate:** If two pages compete for the same intent, redirect the declining page into a stronger sibling.
- **Monitor:** Keep on a watchlist if the drop is recent and may be noise or short-term seasonality.
- **Prune/No Action:** Leave alone or deprecate if the content has low demand.

---

## 2. Risk & Impact Analysis

### Cost of a Wrong Recommendation
In a decision-support system, errors have real costs:

*   **False Positives (Type I Error):** The system recommends refreshing a page that is actually healthy, is undergoing a temporary seasonal dip, or has natural search fluctuations.
    *   *Business Cost:* High. It wastes scarce, expensive editorial hours rewriting content that does not need a refresh, diverting resources from higher-impact pages.
*   **False Negatives (Type II Error):** The system fails to identify a critical high-traffic page in structural decline until its search visibility completely evaporates.
    *   *Business Cost:* Severe. It results in a permanent loss of search discoverability, user traffic, and conversions, which are much harder to recover once competitors occupy those rankings.

---

## 3. Why Data and Machine Learning Can Help

### The Scale Challenge
Human editors cannot monitor daily performance across hundreds of clients and hundreds of thousands of pages. Manual inspection of GSC (Google Search Console) data at this scale is impossible.

### Multi-Dimensional Signals
A page's decline is rarely indicated by a single number. It is a combination of:
- Long-term impression trend vs. short-term click trend.
- Average position decay (e.g., drifting from position 3 to 8).
- CTR dropping below the expected average for that specific position tier.
- Freshness metrics (days since last update, content age).

Machine learning can model the complex, non-linear interactions of these dimensions to rank pages by their likelihood of persistent decline, which is far more accurate than simple one-dimensional filters.

---

## 4. Why This is Not Just "Train a Model"

An operational ML system is not a standalone classifier. Translating a model into real business action requires resolving several system-level problems:

1.  **Defining the Data Contract (Predictive Discipline):** To avoid leakage, we must strictly separate the *feature window* (e.g., first 90 days of signals) from the *target window* (e.g., next 30 days of outcomes). Feeding in post-decision signals (like future clicks) would make the model look accurate in training but render it useless in production.
2.  **Addressing Seasonality and Noise:** A drop in impressions during December for a gardening page is seasonality, not a content quality issue. We must normalize page trends against client-level or topic-level baselines so we do not flag seasonal drops as failures.
3.  **Client-Holdout Splits:** Search behavior and site structure vary widely between clients. If we split pages randomly, the model might overfit to client-specific patterns. We must evaluate the model by holding out entire clients (Group-based validation) to ensure the system generalizes to new sites.
4.  **Explainability (Reason Codes):** Editors will not act on a black-box probability score. The model must output human-readable reason codes explaining *why* a page was surfaced (e.g., "Position dropped by 3 slots while impressions remained high"), ensuring trust and actionable editing.

---

## 5. Cautious Language & Scope Limitations

In alignment with SEO and ML best practices, we emphasize:
- **Decision Support, Not ROI Guarantee:** Surfacing a page for review means there is *evidence of search decline*, not a guarantee that editing the page will cause search rankings to recover. Causal recovery can only be verified via A/B testing or structured experiments.
- **Observational Associations:** Our model identifies statistical associations between pre-decision signals and future traffic drops. We do not claim to have "decoded Google's ranking algorithm" or "uncovered hidden SEO ranking factors."
- **Noise at the Tail:** For low-volume pages (e.g., pages with <100 impressions in 90 days), search metrics are highly volatile and dominated by noise. We will restrict recommendations to pages exceeding minimum exposure thresholds.
