from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score


st.set_page_config(page_title="Hybrid Energy Dashboard", layout="wide")

DATA_PATH = Path("energy_dataset.csv")
BASELINE_MODEL_PATH = Path("power_model_rf.joblib")
FORECAST_MODEL_PATH = Path("power_forecast_1h_rf.joblib")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    out = df.copy()

    hour = ts.dt.hour.to_numpy()
    day_of_year = ts.dt.dayofyear.to_numpy()

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    out["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    return out


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data
def compute_overview_metrics(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "total_energy_kwh": float(df["power_produced_w"].sum() / 1000.0),
        "avg_power_w": float(df["power_produced_w"].mean()),
        "max_power_w": float(df["power_produced_w"].max()),
        "avg_wind_ms": float(df["wind_speed_ms"].mean()),
        "avg_temp_c": float(df["ambient_temp_c"].mean()),
    }


@st.cache_data
def evaluate_models(df: pd.DataFrame) -> dict:
    metrics = {}
    df_feat = add_time_features(df)

    if BASELINE_MODEL_PATH.exists():
        artifact = joblib.load(BASELINE_MODEL_PATH)
        model = artifact["model"]
        features = artifact["feature_cols"]

        y_true = df_feat["power_produced_w"]
        y_pred = model.predict(df_feat[features])

        metrics["baseline_mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["baseline_r2"] = float(r2_score(y_true, y_pred))

    if FORECAST_MODEL_PATH.exists():
        artifact_f = joblib.load(FORECAST_MODEL_PATH)
        model_f = artifact_f["model"]
        features_f = artifact_f["feature_cols"]

        df_fc = df_feat.copy()
        df_fc["power_next_1h_w"] = df_fc["power_produced_w"].shift(-1)
        df_fc = df_fc.dropna(subset=["power_next_1h_w"]).copy()

        split_idx = int(len(df_fc) * 0.8)
        test_df = df_fc.iloc[split_idx:]

        y_true_f = test_df["power_next_1h_w"]
        y_pred_f = model_f.predict(test_df[features_f])

        metrics["forecast_mae"] = float(mean_absolute_error(y_true_f, y_pred_f))
        metrics["forecast_r2"] = float(r2_score(y_true_f, y_pred_f))

    return metrics


def main() -> None:
    st.title("Hybrid Solar and Wind Energy Dashboard")
    st.caption("One-page results view for synthetic digital twin data and ML performance")

    if not DATA_PATH.exists():
        st.error("energy_dataset.csv not found. Run train_energy_model.py first.")
        st.stop()

    df = load_data()
    metrics = compute_overview_metrics(df)
    model_metrics = evaluate_models(df)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rows", f"{metrics['rows']:,}")
    c2.metric("Total Energy", f"{metrics['total_energy_kwh']:.1f} kWh")
    c3.metric("Avg Power", f"{metrics['avg_power_w']:.1f} W")
    c4.metric("Max Power", f"{metrics['max_power_w']:.1f} W")
    c5.metric("Avg Wind", f"{metrics['avg_wind_ms']:.2f} m/s")
    c6.metric("Avg Temp", f"{metrics['avg_temp_c']:.2f} C")

    st.markdown("### Filters")
    start_dt, end_dt = st.slider(
        "Timestamp window",
        min_value=df["timestamp"].min().to_pydatetime(),
        max_value=df["timestamp"].max().to_pydatetime(),
        value=(
            df["timestamp"].min().to_pydatetime(),
            df["timestamp"].max().to_pydatetime(),
        ),
        format="YYYY-MM-DD HH:mm",
    )

    view_df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] <= end_dt)].copy()

    left, right = st.columns(2)

    with left:
        fig_power = px.line(
            view_df,
            x="timestamp",
            y="power_produced_w",
            title="Power Produced Over Time",
        )
        st.plotly_chart(fig_power, use_container_width=True)

        fig_wind = px.histogram(
            view_df,
            x="wind_speed_ms",
            nbins=40,
            title="Wind Speed Distribution",
        )
        st.plotly_chart(fig_wind, use_container_width=True)

    with right:
        fig_lux = px.line(
            view_df,
            x="timestamp",
            y="lux_sensor",
            title="Lux Sensor Over Time",
        )
        st.plotly_chart(fig_lux, use_container_width=True)

        fig_temp = px.line(
            view_df,
            x="timestamp",
            y="ambient_temp_c",
            title="Ambient Temperature Over Time",
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    st.markdown("### Relationship Plots")
    r1, r2 = st.columns(2)

    with r1:
        fig_lux_power = px.scatter(
            view_df,
            x="lux_sensor",
            y="power_produced_w",
            opacity=0.45,
            title="Lux vs Power",
        )
        st.plotly_chart(fig_lux_power, use_container_width=True)

    with r2:
        fig_wind_power = px.scatter(
            view_df,
            x="wind_speed_ms",
            y="power_produced_w",
            opacity=0.45,
            title="Wind Speed vs Power",
        )
        st.plotly_chart(fig_wind_power, use_container_width=True)

    st.markdown("### Model Performance")
    m1, m2, m3, m4 = st.columns(4)

    m1.metric(
        "Baseline MAE",
        f"{model_metrics.get('baseline_mae', float('nan')):.2f} W"
        if "baseline_mae" in model_metrics
        else "Model not found",
    )
    m2.metric(
        "Baseline R2",
        f"{model_metrics.get('baseline_r2', float('nan')):.4f}"
        if "baseline_r2" in model_metrics
        else "Model not found",
    )
    m3.metric(
        "Forecast MAE",
        f"{model_metrics.get('forecast_mae', float('nan')):.2f} W"
        if "forecast_mae" in model_metrics
        else "Model not found",
    )
    m4.metric(
        "Forecast R2",
        f"{model_metrics.get('forecast_r2', float('nan')):.4f}"
        if "forecast_r2" in model_metrics
        else "Model not found",
    )

    st.info(
        "Tip: Run train_energy_model.py before launching this dashboard to ensure data and model artifacts are available."
    )


if __name__ == "__main__":
    main()
