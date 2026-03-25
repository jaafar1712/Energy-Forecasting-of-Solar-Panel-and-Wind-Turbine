# Energy Forecasting of Solar Panel and Wind Turbine

This project builds a synthetic Digital Twin dataset for a hybrid Solar/Wind energy system and trains AI models for power prediction and 1-hour-ahead forecasting.

## Project Highlights

- Generates 1 year of hourly data (8760 rows)
- Simulates realistic sensor behavior (lux, wind speed, wind direction, ambient temperature)
- Computes power, voltage, and current with physics-inspired equations
- Trains baseline and forecast models using Random Forest Regressor
- Saves static plots and supports optional live animated plots
- Exports shareable documentation and flowchart in the docs folder

## Files in This Repository

- [train_energy_model.py](train_energy_model.py): Main pipeline (data generation, plotting, training, evaluation, export)
- [dashboard.py](dashboard.py): One-page interactive dashboard for results visualization
- [docs/energy_model_explanation.md](docs/energy_model_explanation.md): Detailed technical explanation
- [docs/energy_model_explanation.pdf](docs/energy_model_explanation.pdf): Shareable report (PDF)
- [docs/energy_model_flowchart.mmd](docs/energy_model_flowchart.mmd): Mermaid flowchart source
- [scripts/build_explanation_pdf.py](scripts/build_explanation_pdf.py): Script to regenerate the PDF report

## Data Model

### Input Features

- `timestamp`: Hourly datetime
- `lux_sensor`: 0 to 120000 lux (diurnal + seasonal + cloud noise)
- `wind_speed_ms`: 0 to 25 m/s (Weibull distribution)
- `wind_direction_deg`: 0 to 360 degrees
- `ambient_temp_c`: -5 to 35 C (seasonal/daily + lux correlation)

### Output Targets

- `power_produced_w`: Hybrid power output
- `current_produced_a`: Computed as I = P / V
- `voltage_v`: Around 24 V with bounded noise

## Physics-Inspired Relations

- Solar power: proportional to lux
- Wind power: proportional to v^3
- Total power: P_total = P_solar + P_wind
- Current: I = P / V

## Setup

From PowerShell in project root:

```powershell
cd "c:\Users\ACER\solar panel"
.\.venv\Scripts\python.exe -m pip install numpy pandas scikit-learn matplotlib joblib reportlab streamlit plotly
```

## Run the Full Pipeline

### With live plots

```powershell
.\.venv\Scripts\python.exe .\train_energy_model.py
```

### Without live plots

```powershell
.\.venv\Scripts\python.exe .\train_energy_model.py --no-live-plots
```

## One-Page Dashboard

Launch the interactive dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\dashboard.py
```

Dashboard includes:

- KPI cards for energy and power statistics
- Interactive time filtering
- Time-series charts (power, lux, temperature)
- Distribution and correlation-style scatter plots
- Baseline and forecast model performance metrics

## Expected Outputs (Generated Locally)

- `energy_dataset.csv`
- `plots/week_lux_power_profile.png`
- `plots/wind_speed_distribution.png`
- `plots/power_distribution.png`
- `power_model_rf.joblib`
- `power_forecast_1h_rf.joblib`

Note: Large/generated artifacts are intentionally excluded from Git tracking in [.gitignore](.gitignore).

## Model Training Details

### Baseline Model

- Algorithm: RandomForestRegressor
- Task: Predict current-hour `power_produced_w`
- Split: Random 80/20
- Metrics: MAE, R2

### Forecast Model

- Algorithm: RandomForestRegressor
- Task: Predict next-hour power (`power_next_1h_w`)
- Split: Chronological 80/20 (time-aware)
- Metrics: Forecast MAE, Forecast R2

## Regenerate the PDF Report

```powershell
.\.venv\Scripts\python.exe .\scripts\build_explanation_pdf.py
```

Output:

- [docs/energy_model_explanation.pdf](docs/energy_model_explanation.pdf)

## Future Improvements

- Add lag/rolling features for stronger forecasting performance
- Add cyclic encoding for wind direction
- Compare RF with Gradient Boosting/XGBoost
- Replace synthetic data with live Home Assistant data stream
