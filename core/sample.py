"""Synthetic classroom-chair demand for the public walkthrough."""
from io import BytesIO

import numpy as np
import pandas as pd


def sample_series():
    months = pd.date_range("2020-01-01", periods=72, freq="MS")
    t = np.arange(len(months))
    seasonal = np.array([5, 24, 10, -20, -30, -10, 20, 65, 36, 4, -16, -38])
    noise = np.random.default_rng(42).normal(0, 18, len(months))
    units = np.rint(105 + t * 0.7 + seasonal[t % 12] + noise).astype(int)
    return pd.DataFrame({"ds": months, "y": units})


def sample_workbook():
    stream = BytesIO()
    sample_series().rename(columns={"ds": "Date", "y": "Classroom chairs"}).to_excel(
        stream, index=False, engine="openpyxl"
    )
    return stream.getvalue()


def set_series(state, series, product, source):
    """Clear downstream results only when their input actually changes."""
    previous = state.get("ts_df_norm")
    if (state.get("product_name") == product and state.get("series_source") == source
            and previous is not None and previous[["ds", "y"]].equals(series[["ds", "y"]])):
        return
    for key in list(state):
        if key.startswith(("forecast_", "mps_", "experiments_", "demo_", "sens_")) or key in {
            "last_result", "exp_table", "pipeline_experiments", "orders_row_inputs_mps"
        }:
            del state[key]
    state.update(ts_df_norm=series.copy(), product_name=product, upload_ok=True,
                 series_source=source, forecast_committed=False)


def load_sample(state):
    set_series(state, sample_series(), "Classroom chairs", "sample")
