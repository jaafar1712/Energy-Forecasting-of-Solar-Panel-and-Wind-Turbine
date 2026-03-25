import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def generate_synthetic_energy_dataset(
    output_csv: str = "energy_dataset.csv",
    start_date: str = "2025-01-01 00:00:00",
    n_hours: int = 24 * 365,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate one year (8760 rows) of hourly synthetic hybrid solar/wind data."""
    rng = np.random.default_rng(random_seed)

    # 1) Hourly timestamps (non-leap year => 8760 rows)
    timestamp = pd.date_range(start=start_date, periods=n_hours, freq="h")

    hour = timestamp.hour.to_numpy()
    day_of_year = timestamp.dayofyear.to_numpy()

    # 2) Lux model: diurnal sine wave + seasonal variation + cloud noise
    # Daylight profile peaks around noon and is zero at night.
    diurnal = np.sin(np.pi * (hour - 6) / 12)
    diurnal = np.clip(diurnal, 0, None)

    # Seasonal factor for longer/brighter summer days.
    seasonal_lux_factor = 0.65 + 0.35 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    seasonal_lux_factor = np.clip(seasonal_lux_factor, 0.2, 1.0)

    max_lux = 120_000
    clear_sky_lux = max_lux * diurnal * seasonal_lux_factor

    # Cloud attenuation in [0.45, 1.0] and additive sensor noise.
    cloud_factor = np.clip(rng.normal(loc=0.82, scale=0.15, size=n_hours), 0.45, 1.0)
    lux_noise = rng.normal(loc=0.0, scale=2500.0, size=n_hours)

    lux_sensor = clear_sky_lux * cloud_factor + lux_noise
    lux_sensor = np.clip(lux_sensor, 0, max_lux)

    # 3) Wind speed: Weibull distribution clipped to [0, 25] m/s
    wind_shape_k = 2.0
    wind_scale_lambda = 6.5
    wind_speed_ms = rng.weibull(wind_shape_k, size=n_hours) * wind_scale_lambda
    wind_speed_ms = np.clip(wind_speed_ms, 0, 25)

    # 4) Wind direction: 0 to 360 degrees
    wind_direction_deg = rng.uniform(0, 360, size=n_hours)

    # 5) Ambient temperature: seasonal baseline + daily cycle + lux correlation + noise
    seasonal_temp = 13 + 12 * np.sin(2 * np.pi * (day_of_year - 170) / 365)
    daily_temp = 4 * np.sin(2 * np.pi * (hour - 8) / 24)
    lux_correlation = 8 * (lux_sensor / max_lux)
    temp_noise = rng.normal(0, 1.5, size=n_hours)

    ambient_temp_c = seasonal_temp + daily_temp + lux_correlation + temp_noise
    ambient_temp_c = np.clip(ambient_temp_c, -5, 35)

    # 6) Power model (physical intuition)
    # Solar term approximately proportional to irradiance/lux.
    # Wind term approximately proportional to v^3, with clipping for realism.
    solar_rated_w = 2500  # synthetic nominal PV contribution at peak lux
    wind_coeff = 3.2      # synthetic coefficient for v^3 term

    p_solar_w = solar_rated_w * (lux_sensor / max_lux)
    p_wind_w = wind_coeff * np.power(wind_speed_ms, 3)
    p_wind_w = np.clip(p_wind_w, 0, 5000)

    power_produced_w = p_solar_w + p_wind_w

    # 7) Electrical outputs: voltage around 24V +/- 2V, current = power / voltage
    voltage_v = 24 + rng.normal(0, 0.8, size=n_hours)
    voltage_v = np.clip(voltage_v, 22, 26)

    current_produced_a = power_produced_w / voltage_v

    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "lux_sensor": np.round(lux_sensor, 2),
            "wind_speed_ms": np.round(wind_speed_ms, 3),
            "wind_direction_deg": np.round(wind_direction_deg, 2),
            "ambient_temp_c": np.round(ambient_temp_c, 2),
            "power_produced_w": np.round(power_produced_w, 2),
            "current_produced_a": np.round(current_produced_a, 3),
            "voltage_v": np.round(voltage_v, 3),
        }
    )

    df.to_csv(output_csv, index=False)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic hour/day-of-year features suitable for forecasting models."""
    out = df.copy()
    ts = pd.to_datetime(out["timestamp"])

    hour = ts.dt.hour.to_numpy()
    day_of_year = ts.dt.dayofyear.to_numpy()

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    out["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    return out


def save_basic_plots(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """Save quick-look charts for synthetic data sanity checks."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_plot = df.copy()
    df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"])

    # 1-week profile for lux and power
    first_week = df_plot.iloc[: 24 * 7]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(first_week["timestamp"], first_week["lux_sensor"], color="#e6a700", label="Lux")
    ax1.set_ylabel("Lux", color="#e6a700")
    ax1.tick_params(axis="y", labelcolor="#e6a700")

    ax2 = ax1.twinx()
    ax2.plot(first_week["timestamp"], first_week["power_produced_w"], color="#006d77", label="Power")
    ax2.set_ylabel("Power (W)", color="#006d77")
    ax2.tick_params(axis="y", labelcolor="#006d77")

    ax1.set_title("First Week: Lux and Power")
    ax1.set_xlabel("Timestamp")
    fig.tight_layout()
    fig.savefig(out_dir / "week_lux_power_profile.png", dpi=150)
    plt.close(fig)

    # Wind speed distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_plot["wind_speed_ms"], bins=40, color="#1d3557", alpha=0.9)
    ax.set_title("Wind Speed Distribution")
    ax.set_xlabel("Wind Speed (m/s)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "wind_speed_distribution.png", dpi=150)
    plt.close(fig)

    # Target distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_plot["power_produced_w"], bins=45, color="#2a9d8f", alpha=0.9)
    ax.set_title("Power Produced Distribution")
    ax.set_xlabel("Power Produced (W)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_dir / "power_distribution.png", dpi=150)
    plt.close(fig)


def show_live_plots(df: pd.DataFrame, sample_hours: int = 24 * 7, step: int = 3) -> None:
    """Display interactive live plots that update as data points stream in."""
    df_live = df.copy()
    df_live["timestamp"] = pd.to_datetime(df_live["timestamp"])
    n_points = min(sample_hours, len(df_live))
    if n_points < 2:
        return

    ts = df_live["timestamp"].iloc[:n_points].to_numpy()
    lux = df_live["lux_sensor"].iloc[:n_points].to_numpy()
    power = df_live["power_produced_w"].iloc[:n_points].to_numpy()
    wind = df_live["wind_speed_ms"].iloc[:n_points].to_numpy()
    temp = df_live["ambient_temp_c"].iloc[:n_points].to_numpy()

    try:
        plt.ion()

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        ax_lux = axes[0, 0]
        ax_power = axes[0, 1]
        ax_wind = axes[1, 0]
        ax_temp = axes[1, 1]

        (lux_line,) = ax_lux.plot([], [], color="#e6a700", linewidth=1.8)
        ax_lux.set_title("Live Lux Profile")
        ax_lux.set_xlabel("Timestamp")
        ax_lux.set_ylabel("Lux")
        ax_lux.set_xlim(ts[0], ts[-1])
        ax_lux.set_ylim(0, max(1.0, float(np.max(lux)) * 1.05))

        (power_line,) = ax_power.plot([], [], color="#006d77", linewidth=1.8)
        ax_power.set_title("Live Power Output")
        ax_power.set_xlabel("Timestamp")
        ax_power.set_ylabel("Power (W)")
        ax_power.set_xlim(ts[0], ts[-1])
        ax_power.set_ylim(0, max(1.0, float(np.max(power)) * 1.05))

        (wind_line,) = ax_wind.plot([], [], color="#1d3557", linewidth=1.6)
        ax_wind.set_title("Live Wind Speed")
        ax_wind.set_xlabel("Timestamp")
        ax_wind.set_ylabel("Wind Speed (m/s)")
        ax_wind.set_xlim(ts[0], ts[-1])
        ax_wind.set_ylim(0, max(1.0, float(np.max(wind)) * 1.1))

        (temp_line,) = ax_temp.plot([], [], color="#2a9d8f", linewidth=1.6)
        ax_temp.set_title("Live Ambient Temperature")
        ax_temp.set_xlabel("Timestamp")
        ax_temp.set_ylabel("Temp (deg C)")
        ax_temp.set_xlim(ts[0], ts[-1])
        ax_temp.set_ylim(float(np.min(temp)) - 1.0, float(np.max(temp)) + 1.0)

        fig.suptitle("Digital Twin Live Dashboard (First Week Sample)", fontsize=13)
        fig.tight_layout()

        for i in range(2, n_points + 1, max(1, step)):
            lux_line.set_data(ts[:i], lux[:i])
            power_line.set_data(ts[:i], power[:i])
            wind_line.set_data(ts[:i], wind[:i])
            temp_line.set_data(ts[:i], temp[:i])
            fig.canvas.draw_idle()
            plt.pause(0.03)

        plt.ioff()
        plt.show()
    except Exception as exc:
        print(
            "Live plotting is not available in this environment. "
            f"Reason: {exc}. You can still use saved PNG plots in the plots folder."
        )


def train_baseline_power_model(
    csv_path: str = "energy_dataset.csv",
    model_output_path: str = "power_model_rf.joblib",
    random_seed: int = 42,
) -> None:
    """Train a Random Forest model, evaluate, and save model artifacts."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = add_time_features(df)

    feature_cols = [
        "lux_sensor",
        "wind_speed_ms",
        "wind_direction_deg",
        "ambient_temp_c",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]
    target_col = "power_produced_w"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Dataset rows: {len(df)}")
    print(f"Mean Absolute Error (MAE): {mae:.3f} W")
    print(f"R^2 Score: {r2:.4f}")

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "random_seed": random_seed,
    }
    joblib.dump(artifact, model_output_path)
    print(f"Saved trained model artifact to: {model_output_path}")


def train_one_hour_ahead_forecast_model(
    csv_path: str = "energy_dataset.csv",
    model_output_path: str = "power_forecast_1h_rf.joblib",
    random_seed: int = 42,
) -> None:
    """Train a 1-hour-ahead power forecaster and save model artifact."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    df = add_time_features(df)

    # Forecast target: next hour power.
    df["power_next_1h_w"] = df["power_produced_w"].shift(-1)
    df = df.dropna(subset=["power_next_1h_w"]).copy()

    feature_cols = [
        "lux_sensor",
        "wind_speed_ms",
        "wind_direction_deg",
        "ambient_temp_c",
        "hour_sin",
        "hour_cos",
        "doy_sin",
        "doy_cos",
    ]
    target_col = "power_next_1h_w"

    # Time-aware split: first 80% train, last 20% test.
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    model = RandomForestRegressor(
        n_estimators=350,
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("1-hour-ahead forecast model")
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    print(f"Forecast MAE: {mae:.3f} W")
    print(f"Forecast R^2: {r2:.4f}")

    artifact = {
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "horizon_hours": 1,
        "random_seed": random_seed,
    }
    joblib.dump(artifact, model_output_path)
    print(f"Saved 1-hour-ahead model artifact to: {model_output_path}")


def main(live_plots: bool = True) -> None:
    output_csv = "energy_dataset.csv"
    plot_dir = "plots"
    model_path = "power_model_rf.joblib"
    forecast_model_path = "power_forecast_1h_rf.joblib"

    df = generate_synthetic_energy_dataset(output_csv=output_csv)
    print(f"Generated {output_csv} with shape: {df.shape}")
    save_basic_plots(df, output_dir=plot_dir)
    print(f"Saved plots to folder: {plot_dir}")

    if live_plots:
        print("Opening live plots window...")
        show_live_plots(df)

    train_baseline_power_model(csv_path=output_csv, model_output_path=model_path)
    train_one_hour_ahead_forecast_model(
        csv_path=output_csv,
        model_output_path=forecast_model_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-live-plots",
        action="store_true",
        help="Disable interactive live plot windows.",
    )
    args = parser.parse_args()
    main(live_plots=not args.no_live_plots)
