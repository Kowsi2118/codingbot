from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW


def _weighted_mean_and_se(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    stats = DescrStatsW(values, weights=weights, ddof=1)
    mean = float(stats.mean)
    se = float(stats.std_mean)
    return mean, se


def _weighted_proportion_and_se(
    values: np.ndarray, weights: np.ndarray, success_value
) -> Tuple[float, float]:
    indicator = (values == success_value).astype(float)
    stats = DescrStatsW(indicator, weights=weights, ddof=1)
    p = float(stats.mean)
    # Approx standard error for proportion under weights using DescrStatsW
    se = float(stats.std_mean)
    return p, se


def compute_weighted_summaries(
    df: pd.DataFrame,
    weight_column: Optional[str],
    measures: Dict[str, Dict],
    groupby: Optional[List[str]] = None,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """
    measures example:
    {
      "income": {"type": "mean"},
      "employed": {"type": "proportion", "success_value": 1}
    }
    """
    if not measures:
        raise ValueError("measures configuration is required")

    if groupby is None or len(groupby) == 0:
        group_keys = ["__ALL__"]
        df = df.copy()
        df["__ALL__"] = "ALL"
    else:
        group_keys = groupby

    z = 1.96 if confidence_level == 0.95 else 1.96

    results = []

    for keys, group in df.groupby(group_keys):
        if not isinstance(keys, tuple):
            keys = (keys,)

        weights = (
            group[weight_column].values.astype(float)
            if weight_column and weight_column in group.columns
            else np.ones(len(group), dtype=float)
        )

        for var, cfg in measures.items():
            mtype = cfg.get("type", "mean").lower()

            if var not in group.columns:
                continue

            values = group[var].values.astype(float)

            if mtype == "mean":
                est, se = _weighted_mean_and_se(values, weights)
            elif mtype == "proportion":
                success_value = cfg.get("success_value", 1)
                est, se = _weighted_proportion_and_se(values, weights, success_value)
            elif mtype == "sum":
                # Weighted sum and its SE via variance of sum = var(mean)*n^2 when weights considered is non-trivial;
                # here we approximate by sum(weights * values) and use std of mean * n as SE.
                stats = DescrStatsW(values, weights=weights, ddof=1)
                est = float(np.sum(values * weights))
                se = float(stats.std_mean * len(values))
            else:
                continue

            moe = z * se

            row = {"measure": var, "type": mtype, "estimate": est, "se": se, "moe": moe}

            for idx, key in enumerate(group_keys):
                row[key] = keys[idx]

            results.append(row)

    return pd.DataFrame(results)