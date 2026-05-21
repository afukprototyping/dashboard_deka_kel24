# Kelompok 24 - Customer Experience Dashboard Always Healthy Hospital

Streamlit dashboard for visualising patient satisfaction survey data across 15 hospital branches in Indonesia (Jan–Dec 2025, 3,600 respondents).

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Features

| Feature | Description |
|---|---|
| **KPI Metrics** | NPS, CSI, CLI, CES with delta vs overall average |
| **Filters** | Branch, Gender, Age Group, Date Range, Touchpoints |
| **Area Analysis** | Branch ranking table, NPS bar chart, touchpoint heatmap, worst-area alerts |
| **Demographics** | KPIs by gender & age group, promoter % breakdown, age distribution |
| **Time Series** | Monthly/quarterly NPS + CSI/CLI/CES trends, top-5 vs bottom-5 branch trends |
| **Touchpoints** | Radar chart, correlation with NPS, word cloud, complaint frequency |
| **AI Insights** | Claude-powered executive summary, branch improvement plan, action plans |
| **Export** | Download filtered data as CSV, download AI insights as TXT |

## Data

- `data.csv` — survey data (semicolon-delimited)
- Columns: `Datetime`, `Branch`, `Gender`, `Age`, `NPS` (0–10), `CSI` (1–5), `Loyalty` (1–5), `CES` (1–5), 10 touchpoint scores, `Improvement_Feedback`

## KPI Formulas

- **NPS** = (Promoters% − Detractors%) × 100, where Promoters = NPS 9–10, Detractors = NPS 0–6
- **CSI** = Mean of CSI score (1–5)
- **CLI** = Mean of Loyalty score (1–5)
- **CES** = Mean of CES score (1–5); *lower is better*
