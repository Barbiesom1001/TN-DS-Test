# Customer Segmentation & Personalized Marketing
### Data Science / AI Solution Proposal — SME Retail

> **Proposal for:** Data Science Internship Assignment  
> **Domain:** Retail  
> **Goal:** Maximize revenue by targeting the right customer with the right promotion

---

## Business Problem

SME Retail businesses typically run **one-size-fits-all promotions** — giving the same discount to every customer regardless of their behavior. This causes:

| Problem | Impact |
|---|---|
| No customer segmentation | Budget wasted on customers who'd buy anyway |
| Promotion margin leakage | Discount given without incremental revenue |
| No churn early warning | React too late when customers are already gone |

**KPIs tracked:** Revenue Per Customer (RPC) · Promotion ROI · Customer Retention Rate

---

## Proposed Solution

**RFM Segmentation + K-Means Clustering + Personalized Offer Engine**

```
Data Input → RFM Features → K-Means Clustering → Segment Profiling → Offer Engine → Measure
```

| Component | Detail |
|---|---|
| **Input** | Sales transactions, Customer master, Product master, Promotion master |
| **Model** | K-Means clustering on scaled RFM features (k=5) |
| **Output** | Customer segment label + recommended promotion action per segment |
| **User** | Marketing team — decides which promotion to run for each group |

**Why AI over rule-based / dashboard?**
- Discovers hidden patterns without pre-defined rules
- Scales to tens of thousands of customers automatically
- Output is an actionable recommendation, not just a chart

---

## Repository Structure

```
├── customer_segmentation_notebook.py   # Main notebook (paste into Colab)
├── rfm_output.csv                      # Mock output: customer → segment mapping
├── eda_overview.png                    # EDA charts (revenue by category, trend, frequency dist)
├── elbow_silhouette.png                # K selection: Elbow + Silhouette Score
├── clusters_scatter.png                # Cluster visualization (Recency vs Monetary)
├── dashboard_mock.png                  # Mock dashboard: segment size + avg revenue
└── README.md                           # This file
```

> **Slide Proposal** (`.pptx`) submitted separately via the submission form.

---

## Data Plan

### Tables Used

| Table | Key Columns | Used For |
|---|---|---|
| `sales` | datetime, customer_id, price, qty, promotion_id | Compute R, F, M |
| `customer_master` | customer_id, customer_taxonomies | Enrich segment with demographics |
| `product_master` | product_id, price, product_taxonomies | Category preference per segment |
| `promotion_master` | promotion_id, discount, start/end_date | Promotion ROI analysis |

### Join Strategy
```
sales
  LEFT JOIN customer_master  ON customer_id
  LEFT JOIN product_master   ON product_id
  LEFT JOIN promotion_master ON promotion_id
→ enriched_sales → RFM computation
```

### Data Quality Issues
- **Missing `customer_id` (~5%)** → treat as guest, exclude from clustering
- **Duplicate transactions** → deduplicate on `(datetime, product_id, customer_id)`
- **Cold-start customers** (< 3 transactions) → exclude from K-Means, handle separately

### Mock Data Assumptions
- 500 customers, 12 months (Jan–Dec 2024), 4 product categories
- Purchase frequency simulated per true segment type (High-Value: 15–25 txn, Churned: 1–3 txn)
- ~30% of transactions include a promotion
- Added `_true_segment` column (ground truth) for evaluation only — not used in model

---

## Methodology

### 1. RFM Feature Engineering
| Feature | Definition | Business Meaning |
|---|---|---|
| **Recency** | Days since last purchase | Lower = more engaged |
| **Frequency** | # unique orders (po_id) | Higher = more loyal |
| **Monetary** | Total spend (THB) | Higher = more valuable |

### 2. K-Means Clustering
- Features scaled with `StandardScaler` before clustering
- Optimal k selected via **Elbow method** + **Silhouette Score**
- Chosen **k = 5** (Silhouette ≈ 0.41 on mock data)

### 3. Segment Naming & Promotion Mapping

| Segment | Profile | Action | Channel | Discount |
|---|---|---|---|---|
| Champions | High F, High M, Low R | VIP Early Access | Push Notification | 5–10% |
| Loyal Customers | High F, Medium R | Cross-sell Bundle | Email | 10–15% |
| Potential Growth | Mid F, Mid M | Engagement Promo | Line OA | 15% |
| New / Occasional | Low F, Low R | Welcome Series | SMS | 20% |
| At Risk / Churned | High R (dormant) | Win-back Campaign | Email + SMS | 30%+ |

---

## Validation Approach

**Business Metrics**
- Revenue Per Customer before vs after targeted promotion
- Promo ROI = Revenue gained / Promotion cost
- Retention rate at 30 / 60 / 90 days

**Technical Metrics**
- Silhouette Score > 0.45 (target for real data)
- Inertia / Elbow stability
- Segment stability across time periods (month-over-month)

**Validate without full model**
- RFM rule-based baseline vs K-Means clustering
- A/B test: targeted promo group vs broadcast group
- Business team review: "Does this segment make sense to you?"

---

## Roadmap

| Phase | Timeline | Activities |
|---|---|---|
| **1. Discover** | Week 1–2 | Data audit, EDA, stakeholder interviews, KPI baseline |
| **2. Build** | Week 3–5 | RFM engineering, K-Means, segment naming & profiling |
| **3. Validate** | Week 6–7 | A/B test design, pilot promo per segment, business review |
| **4. Scale** | Week 8+ | Dashboard deployment, automation pipeline, monthly re-clustering |

**MVP (segment + pilot promotion) deliverable within 7 weeks.**

---

## AI Tools Used

| Tool | How Used | How Verified |
|---|---|---|
| Claude (Anthropic) | Code scaffolding, business logic review, structure design, segment naming logic | All code traced line-by-line; distributions validated against SME benchmarks; charts visually inspected |

> All AI-generated outputs were reviewed and verified before inclusion. Segment naming rules and promotion mapping logic were cross-checked with business rationale.

---

## How to Run

**Option 1: Google Colab**
1. Open [Google Colab](https://colab.research.google.com)
2. Upload `customer_segmentation_notebook.py`
3. Run all cells — no additional setup needed (uses standard libraries)

**Option 2: Local**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python customer_segmentation_notebook.py
```

**Output files generated:** 
- `eda_overview.png` — EDA charts
- `elbow_silhouette.png` — K selection
- `clusters_scatter.png` — Cluster visualization
- `dashboard_mock.png` — Mock dashboard
- `rfm_output.csv` — Customer → segment mapping table
