# SEIR physics-informed neural network
Python 3.10, TensorFlow 2.15 (`pip install -r requirements.txt`). Run scripts from their own folder.

## Data
- `scripts/real_data_processing/data_processing.py` - UKHSA COVID-19 cases -> `data/*_data_study.npy`
- `scripts/real_data_processing/deaths_hospitalisations.py` - UKHSA hospitalisations and deaths
- `scripts/synthetic_data_generation/SEIR_model.py` - constant-beta SEIR data (and the report's SEIR panel)
- `scripts/synthetic_data_generation/SEIR_Gaussian_noise.py` - noise scenarios (and the report's noise panel)
- `scripts/synthetic_data_generation/metapopulation_SEIR.py` - 5-patch metapopulation data

## Synthetic-data PINN (`scripts/PINN_files`)
- `PINN_simulated_data.py` - PINN with sigma and gamma provided; also the PINN used by every script below
- `PINN_all_learnable_parameters.py` - beta, sigma and gamma learnt from I(t) only
- `synthetic_scenarios.py` - all synthetic scenarios (constant, spline, noise, forecastable)
- `run_rolling.py` - rolling-origin fits for the conformal intervals
- `forecast_origins.py`, `forecast_origins_figures.py` - forecast skill by origin and horizon
- `conformal_synthetic.py`, `conformal_rolling.py`, `conformal_rolling_figures.py` - causal conformal intervals
- `panel_figures.py`, `synthetic_data_figures.py`, `R_recovery.py` - report figures and tables
- `collect_report_figures.py` - copies every report figure into `report_figures_final/`

## Real-data PINN
- `PINN_hospitalisations_deaths.py` - SEIR-PINN on England cases, hospitalisations and deaths -> `forecasts/PINN.pkl`
- `conformal_PINN.py` - conformal intervals for its forecasts
- `scripts/baseline_models/` - LSTM, Transformer and Prophet baselines -> `forecasts/preds_*.pkl`
- `compare_forecasts.py`, `real_data_figures.py` - model comparison and real-data report figures

## Metapopulation
- `PINN_metapopulation.py` - PINN on the 5-patch metapopulation SEIR model
