# AI-Enhanced Survey Processing Prototype

An AI-augmented Streamlit app to ingest survey files, clean data (imputation, outlier & rule-based checks), apply weights, compute estimates with margins of error, run natural-language SQL Q&A (DuckDB), generate dashboards, and export standardized HTML/PDF reports.

## Features
- Upload CSV/Excel, schema hints via JSON rules
- Cleaning: mean/median/KNN imputation, IQR/Z-score outliers, winsorization, rule-based validation
- Weights: apply design weights and compute weighted means/proportions/sums with MOE
- SQL Q&A: Natural language to SQL (OpenAI optional) and SQL runner using DuckDB
- Dashboard: Quick chart builder and optional NL-to-Plotly suggestions
- Reports: HTML report (PDF optional with wkhtmltopdf)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional for PDF export:
- Install `wkhtmltopdf` binary on your system so `pdfkit` can generate PDFs.

## Run
```bash
streamlit run streamlit_app.py
```

Provide your OpenAI API key in the left sidebar if you want NL-to-SQL or NL dashboard suggestions.

## Notes
- Weighted estimates use `statsmodels` DescrStatsW for estimates and SE; MOE at 95% uses z=1.96.
- SQL runs against an in-memory DuckDB with the cleaned DataFrame registered as `data`.
- This is a prototype; adapt rule formats and estimation modules for your methodology.
