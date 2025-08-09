from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer


@dataclass
class CleaningLog:
    messages: List[str]

    def log(self, message: str) -> None:
        self.messages.append(message)


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    types: Dict[str, str] = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        else:
            types[col] = "categorical"
    return types


def impute_missing(
    df: pd.DataFrame,
    method: str = "median",
    knn_k: int = 5,
    log: Optional[CleaningLog] = None,
) -> pd.DataFrame:
    method = method.lower()
    if log:
        log.log(f"Imputation method: {method}")

    result = df.copy()
    column_types = infer_column_types(result)

    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in column_types.items() if t == "categorical"]

    # Numeric imputation
    if numeric_cols:
        if method == "mean":
            result[numeric_cols] = result[numeric_cols].apply(
                lambda s: s.fillna(s.mean())
            )
            if log:
                log.log(f"Filled NA in numeric columns with mean: {numeric_cols}")
        elif method == "median":
            result[numeric_cols] = result[numeric_cols].apply(
                lambda s: s.fillna(s.median())
            )
            if log:
                log.log(f"Filled NA in numeric columns with median: {numeric_cols}
")
        elif method == "knn":
            imputer = KNNImputer(n_neighbors=knn_k)
            result[numeric_cols] = imputer.fit_transform(result[numeric_cols])
            if log:
                log.log(f"Applied KNN imputation (k={knn_k}) to: {numeric_cols}")
        else:
            raise ValueError("Unknown imputation method for numeric columns")

    # Categorical imputation: fill with mode
    for c in categorical_cols:
        mode_val = result[c].mode(dropna=True)
        if len(mode_val) > 0:
            result[c] = result[c].fillna(mode_val.iloc[0])
            if log:
                log.log(f"Filled NA in categorical column '{c}' with mode")
        else:
            result[c] = result[c].fillna("UNKNOWN")
            if log:
                log.log(f"Filled NA in categorical column '{c}' with 'UNKNOWN'")

    return result


def detect_outliers_iqr(
    s: pd.Series, multiplier: float = 1.5
) -> pd.Series:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (s < lower) | (s > upper)


def detect_outliers_zscore(
    s: pd.Series, threshold: float = 3.0
) -> pd.Series:
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=s.index)
    z = (s - mean) / std
    return z.abs() > threshold


def winsorize_series(
    s: pd.Series, lower_quantile: float = 0.01, upper_quantile: float = 0.99
) -> pd.Series:
    lower = s.quantile(lower_quantile)
    upper = s.quantile(upper_quantile)
    return s.clip(lower=lower, upper=upper)


def handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    winsorize: bool = False,
    winsor_limits: Tuple[float, float] = (0.01, 0.99),
    log: Optional[CleaningLog] = None,
) -> pd.DataFrame:
    method = method.lower()
    result = df.copy()
    numeric_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]

    for col in numeric_cols:
        if method == "iqr":
            mask = detect_outliers_iqr(result[col], multiplier=iqr_multiplier)
        elif method == "zscore":
            mask = detect_outliers_zscore(result[col], threshold=z_threshold)
        else:
            raise ValueError("Unknown outlier detection method")

        outlier_count = int(mask.sum())
        if winsorize and outlier_count > 0:
            lq, uq = winsor_limits
            result.loc[mask, col] = winsorize_series(result[col], lq, uq)[mask]
            if log:
                log.log(
                    f"Winsorized {outlier_count} outliers in '{col}' with limits {winsor_limits}"
                )
        else:
            # Set outliers to NA to be handled by imputation
            result.loc[mask, col] = np.nan
            if log and outlier_count > 0:
                log.log(f"Set {outlier_count} outliers to NA in '{col}'")

    return result


def apply_rules(
    df: pd.DataFrame,
    rules: List[Dict],
    log: Optional[CleaningLog] = None,
) -> pd.DataFrame:
    """
    Rules format (list of dicts):
    {
        "condition": "age < 0",
        "action": "set_null",  # or "cap", "floor", "set_value"
        "columns": ["age"],
        "value": 0  # for set_value/cap/floor
    }
    """
    result = df.copy()
    for rule in rules or []:
        condition = rule.get("condition")
        action = rule.get("action", "set_null").lower()
        columns = rule.get("columns", [])
        value = rule.get("value")

        if not condition or not columns:
            continue

        try:
            mask = result.eval(condition)
        except Exception:
            # Fallback to query-like using pandas eval safety
            try:
                mask = result.query(condition).index
                mask = result.index.isin(mask)
            except Exception:
                if log:
                    log.log(f"Failed to evaluate rule condition: {condition}")
                continue

        affected = int(mask.sum()) if hasattr(mask, "sum") else int(mask.size)

        for col in columns:
            if action == "set_null":
                result.loc[mask, col] = np.nan
            elif action == "cap":
                if value is not None:
                    result.loc[mask, col] = np.minimum(result.loc[mask, col], value)
            elif action == "floor":
                if value is not None:
                    result.loc[mask, col] = np.maximum(result.loc[mask, col], value)
            elif action == "set_value":
                result.loc[mask, col] = value

        if log:
            log.log(f"Applied rule '{action}' on {columns} for {affected} rows")

    return result


def run_cleaning_pipeline(
    df: pd.DataFrame,
    imputation_method: str = "median",
    outlier_method: str = "iqr",
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    winsorize_flag: bool = False,
    winsor_limits: Tuple[float, float] = (0.01, 0.99),
    rules: Optional[List[Dict]] = None,
) -> Tuple[pd.DataFrame, CleaningLog]:
    log = CleaningLog(messages=[])

    # Apply rule-based validation first to mark/correct invalids
    df_rules = apply_rules(df, rules or [], log)

    # Handle outliers
    df_out = handle_outliers(
        df_rules,
        method=outlier_method,
        z_threshold=z_threshold,
        iqr_multiplier=iqr_multiplier,
        winsorize=winsorize_flag,
        winsor_limits=winsor_limits,
        log=log,
    )

    # Impute missing values
    df_imp = impute_missing(df_out, method=imputation_method, log=log)

    return df_imp, log