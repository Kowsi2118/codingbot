from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.processing import run_cleaning_pipeline, CleaningLog
from app.weights import compute_weighted_summaries
from app.sql_nl import generate_sql_from_question, run_sql
from app.reporting import render_report_html, html_to_pdf_bytes


st.set_page_config(page_title="AI Survey Processor", layout="wide")


@dataclass
class CleaningConfig:
    imputation_method: str = "median"
    outlier_method: str = "iqr"
    z_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    winsorize_flag: bool = False
    winsor_limits_low: float = 0.01
    winsor_limits_high: float = 0.99


def _read_upload(upload) -> pd.DataFrame:
    if upload.name.lower().endswith(".csv"):
        return pd.read_csv(upload)
    else:
        return pd.read_excel(upload)


def _download_button(data: bytes, filename: str, mime: str, label: str):
    b64 = base64.b64encode(data).decode()
    href = f'<a download="{filename}" href="data:{mime};base64,{b64}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)


if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "clean_df" not in st.session_state:
    st.session_state.clean_df = None
if "logs" not in st.session_state:
    st.session_state.logs = []


st.title("AI-Enhanced Survey Data Preparation, Estimation, SQL Q&A and Dashboards")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("OpenAI API Key (optional)", type="password")
    st.caption("Used for Natural Language to SQL and Dashboard generation")

    st.markdown("---")
    st.subheader("Cleaning")
    imp_method = st.selectbox("Imputation", ["mean", "median", "knn"], index=1)
    out_method = st.selectbox("Outlier detection", ["iqr", "zscore"], index=0)
    z_thr = st.number_input("Z-score threshold", value=3.0)
    iqr_mult = st.number_input("IQR multiplier", value=1.5)
    wins = st.checkbox("Winsorize outliers before imputation", value=False)
    wlow = st.number_input("Winsor low quantile", value=0.01, min_value=0.0, max_value=0.49)
    whigh = st.number_input("Winsor high quantile", value=0.99, min_value=0.51, max_value=1.0)

    st.markdown("---")
    st.subheader("Weights & Estimation")
    conf_level = st.selectbox("Confidence level", [0.95], index=0)

    config = CleaningConfig(
        imputation_method=imp_method,
        outlier_method=out_method,
        z_threshold=z_thr,
        iqr_multiplier=iqr_mult,
        winsorize_flag=wins,
        winsor_limits_low=wlow,
        winsor_limits_high=whigh,
    )


tabs = st.tabs([
    "1) Upload & Schema", "2) Cleaning", "3) Weights & Estimates", "4) SQL Q&A", "5) Dashboard", "6) Report"
])

with tabs[0]:
    st.header("Upload Data")
    upload = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"]) 
    rules_json = st.text_area(
        "Optional rule-based validations (JSON)",
        value='[\n  {"condition": "age < 0", "action": "set_null", "columns": ["age"]}\n]',
        height=120,
    )

    if upload is not None:
        raw_df = _read_upload(upload)
        st.session_state.raw_df = raw_df
        st.success(f"Loaded data with shape {raw_df.shape}")
        st.dataframe(raw_df.head(20))

    st.caption("Schema mapping can be specified via rules or used implicitly. More advanced mapping can be added later.")

with tabs[1]:
    st.header("Cleaning")
    if st.session_state.raw_df is None:
        st.info("Upload data in the previous tab.")
    else:
        try:
            rules = json.loads(rules_json) if rules_json.strip() else []
        except Exception:
            st.warning("Invalid rules JSON. Ignoring rules.")
            rules = []

        clean_df, log = run_cleaning_pipeline(
            st.session_state.raw_df,
            imputation_method=config.imputation_method,
            outlier_method=config.outlier_method,
            z_threshold=config.z_threshold,
            iqr_multiplier=config.iqr_multiplier,
            winsorize_flag=config.winsorize_flag,
            winsor_limits=(config.winsor_limits_low, config.winsor_limits_high),
            rules=rules,
        )
        st.session_state.clean_df = clean_df
        st.session_state.logs = log.messages
        st.success("Cleaning complete")
        st.dataframe(clean_df.head(20))

        buf = io.BytesIO()
        clean_df.to_csv(buf, index=False)
        _download_button(buf.getvalue(), "cleaned.csv", "text/csv", "Download cleaned CSV")

with tabs[2]:
    st.header("Weights & Estimation")
    if st.session_state.clean_df is None:
        st.info("Run cleaning first.")
    else:
        df = st.session_state.clean_df
        cols = list(df.columns)
        weight_col = st.selectbox("Weight column (optional)", ["<none>"] + cols)
        groupby_cols = st.multiselect("Group by columns (optional)", cols, default=[])
        measure_cols = st.multiselect("Measure variables", [c for c in cols if pd.api.types.is_numeric_dtype(df[c])])

        measures_cfg: Dict[str, Dict] = {}
        for c in measure_cols:
            mtype = st.selectbox(f"Type for {c}", ["mean", "sum", "proportion"], key=f"type_{c}")
            cfg: Dict = {"type": mtype}
            if mtype == "proportion":
                success_value = st.number_input(f"Success value for {c}", value=1.0, key=f"succ_{c}")
                cfg["success_value"] = success_value
            measures_cfg[c] = cfg

        if st.button("Compute estimates"):
            est = compute_weighted_summaries(
                df,
                weight_column=None if weight_col == "<none>" else weight_col,
                measures=measures_cfg,
                groupby=groupby_cols,
                confidence_level=conf_level,
            )
            st.dataframe(est)
            st.session_state.estimates = est

with tabs[3]:
    st.header("SQL Q&A (DuckDB)")
    if st.session_state.clean_df is None:
        st.info("Run cleaning first.")
    else:
        df = st.session_state.clean_df
        st.write("Ask a question in natural language. The app will propose a SQL query and run it.")
        q = st.text_input("Question", value="What is the average of the first numeric column?")
        if st.button("Generate SQL"):
            sql = generate_sql_from_question(q, df, openai_api_key=api_key)
            st.code(sql, language="sql")
            try:
                out = run_sql(sql, df)
                st.dataframe(out)
            except Exception as e:
                st.error(str(e))

        st.markdown("### Or run your own SQL")
        user_sql = st.text_area("SQL", value="SELECT * FROM data LIMIT 10;")
        if st.button("Run SQL"):
            try:
                out = run_sql(user_sql, df)
                st.dataframe(out)
            except Exception as e:
                st.error(str(e))

with tabs[4]:
    st.header("Dashboard")
    if st.session_state.clean_df is None:
        st.info("Run cleaning first.")
    else:
        df = st.session_state.clean_df
        st.write("Create a quick chart or ask a dashboard question.")

        # Manual quick chart
        st.subheader("Quick Chart Builder")
        xcol = st.selectbox("X-axis", df.columns)
        ycol = st.selectbox("Y-axis (numeric)", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
        color = st.selectbox("Color (optional)", ["<none>"] + list(df.columns))
        chart_type = st.selectbox("Chart type", ["bar", "line", "scatter", "box"]) 
        if st.button("Render chart"):
            kwargs = {}
            if color != "<none>":
                kwargs["color"] = color
            if chart_type == "bar":
                fig = px.bar(df, x=xcol, y=ycol, **kwargs)
            elif chart_type == "line":
                fig = px.line(df, x=xcol, y=ycol, **kwargs)
            elif chart_type == "scatter":
                fig = px.scatter(df, x=xcol, y=ycol, **kwargs)
            else:
                fig = px.box(df, x=xcol, y=ycol, **kwargs)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Ask with Natural Language (optional)")
        st.caption("If an API key is provided, the model will suggest a chart using Plotly over the cleaned DataFrame variable named df.")
        prompt = st.text_input("Question", value="Show average income by state as a bar chart")
        if st.button("Suggest dashboard"):
            if not api_key:
                st.warning("Provide an OpenAI API key in the sidebar to use NL dashboard suggestion.")
            else:
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=api_key)
                    sys = (
                        "You are a helpful assistant that writes Python code to create Plotly Express charts. "
                        "The cleaned pandas DataFrame is available as a variable named df. Return only Python code that defines a variable 'fig'."
                    )
                    user = f"Create a single Plotly Express figure for this request: {prompt}. Use only columns present in df."
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "system", "content": sys}, {"role": "user", "content": user}],
                        temperature=0,
                    )
                    code = resp.choices[0].message.content.strip()
                    if code.startswith("```) ") or code.startswith("```"):
                        code = code.strip("`\n ")
                    local_vars: Dict = {"px": px, "df": df}
                    exec(code, {}, local_vars)
                    fig = local_vars.get("fig")
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("The model did not produce a 'fig'.")
                except Exception as e:
                    st.error(str(e))

with tabs[5]:
    st.header("Report Generation")
    if st.session_state.clean_df is None:
        st.info("Run cleaning first.")
    else:
        df = st.session_state.clean_df
        est = st.session_state.get("estimates")

        # Prepare charts as PNGs to embed in report
        charts = []
        try:
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                fig = px.histogram(df, x=num_cols[0])
                buf = io.BytesIO()
                fig.write_image(buf, format="png")
                charts.append({
                    "title": f"Distribution of {num_cols[0]}",
                    "png_base64": base64.b64encode(buf.getvalue()).decode(),
                })
        except Exception:
            # plotly image export may need 'kaleido' installed; fallback to no charts
            pass

        config_dict = {
            "imputation_method": config.imputation_method,
            "outlier_method": config.outlier_method,
            "winsorize_flag": config.winsorize_flag,
        }

        estimates_rows = []
        estimates_columns = []
        if est is not None and isinstance(est, pd.DataFrame) and not est.empty:
            estimates_columns = list(est.columns)
            estimates_rows = est.fillna("").astype(str).to_dict(orient="records")

        context = {
            "title": "Official Survey Processing Report",
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "config": config_dict,
            "logs": st.session_state.logs,
            "estimates": estimates_rows,
            "estimates_columns": estimates_columns,
            "charts": charts,
        }

        html = render_report_html(context)
        st.download_button("Download HTML Report", data=html, file_name="report.html")

        pdf_bytes = html_to_pdf_bytes(html)
        if pdf_bytes is not None:
            st.download_button("Download PDF Report", data=pdf_bytes, file_name="report.pdf")
        else:
            st.info("PDF generation requires 'pdfkit' and the wkhtmltopdf system binary. Only HTML is available now.")

st.markdown("---")
with st.expander("Help & Notes"):
    st.markdown(
        """
        - Upload CSV/Excel and configure cleaning. Rules accept simple conditions like `age < 0` with actions.
        - Apply design weights by selecting the weight column and computing summaries.
        - SQL Q&A uses DuckDB; Natural language requires an OpenAI API key. Table is named `data`.
        - Dashboard tab supports manual chart builder and optional NL suggestion.
        - Reports can be downloaded as HTML; PDF requires `wkhtmltopdf` installed on the server.
        """
    )