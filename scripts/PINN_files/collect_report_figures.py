import os
import shutil

### Copy the report figures into one folder, report_figures_final/, with a numbered subfolder per section and
### the figures numbered in report order (each subfolder is rebuilt from scratch every run). Forecast figures
### carry causal rolling-origin conformal intervals (each scenario calibrated on its own earlier errors).
### Run after panel_figures.py, forecast_origins_figures.py, conformal_rolling_figures.py (provided and learned),
### synthetic_data_figures.py and R_recovery.py. real_data_figures.py writes 4_real_data itself.

src = "../../report_figures"
dst = "../../report_figures_final"
CR = "conformal_rolling/rolling_conformal"
CRL = "conformal_rolling_learned/rolling_conformal"
RES = "../../results/PINN_simulated_data"

### the synthetic SEIR data (time-varying: the five scenarios the forecasts use)
SYNTHETIC_DATA = [
    ("fig02_SEIR_constant_beta", "SEIR_constant_beta"),
    ("fig02_SEIR_constant_beta_0.75", "SEIR_constant_beta_0.75"),          ### synthetic_data_figures.py
    ("fig07_SEIR_time_varying_beta_forecast_scenarios", "SEIR_time_varying_beta"),
    ("fig12_SEIR_gaussian_noise", "SEIR_gaussian_noise"),
]

### sigma and gamma provided to the PINN
PARAMS_PROVIDED = [
    (f"{CR}_forecast_constant_80", "forecast_constant_beta"),
    (f"{CR}_forecast_origins_constant", "forecast_origins_constant_beta"),
    ("fig05_beta_recon_constant_beta_params_provided", "beta_recon_constant_beta"),
    (f"{CR}_forecast_time_varying_80", "forecast_time_varying_beta"),
    (f"{CR}_forecast_origins_time_varying", "forecast_origins_time_varying_beta"),
    ("fig10_beta_recon_time_varying_beta_params_provided", "beta_recon_time_varying_beta"),
    (f"{CR}_forecast_forecastable_80", "forecast_forecastable"),
    (f"{CR}_forecast_origins_forecastable", "forecast_origins_forecastable"),
    ("forecast_origins/beta_origins_forecastable_provided", "beta_extrapolation_forecastable"),
    ("forecast_origins/forecast_error_by_horizon_provided", "forecast_error_by_horizon"),
    ("forecast_origins/beta_error_by_horizon_provided", "beta_error_by_horizon"),
    (f"{CR}_forecast_noise_80", "forecast_noise"),
    (f"{CR}_forecast_origins_noise", "forecast_origins_noise"),
    ("fig15_beta_recon_noise_params_provided", "beta_recon_noise"),
    (f"{CR}_coverage_by_horizon_aci", "conformal_coverage_by_horizon_aci"),
    (f"{CR}_coverage_by_horizon_split", "conformal_coverage_by_horizon_split"),
]

### sigma and gamma learned from the infection data only; R(t) = beta/gamma next to the beta reconstruction
INFECTION_DATA_ONLY = [
    (f"{CRL}_forecast_constant_80", "forecast_constant_beta"),
    (f"{CRL}_forecast_origins_constant", "forecast_origins_constant_beta"),
    ("fig06_beta_recon_constant_beta_infection_data_only", "beta_recon_constant_beta"),
    (f"{CRL}_forecast_time_varying_80", "forecast_time_varying_beta"),
    (f"{CRL}_forecast_origins_time_varying", "forecast_origins_time_varying_beta"),
    ("fig11_beta_recon_time_varying_beta_infection_data_only", "beta_recon_time_varying_beta"),
    ("R_recovery/R_recon_time_varying_infection_data_only", "R_recon_time_varying_beta"),
    ("R_recovery/R_vs_beta_error_infection_data_only", "R_vs_beta_error"),
    (f"{CRL}_forecast_forecastable_80", "forecast_forecastable"),
    (f"{CRL}_forecast_origins_forecastable", "forecast_origins_forecastable"),
    ("forecast_origins/beta_origins_forecastable_learned", "beta_extrapolation_forecastable"),
    ("forecast_origins/forecast_error_by_horizon_learned", "forecast_error_by_horizon"),
    ("forecast_origins/beta_error_by_horizon_learned", "beta_error_by_horizon"),
    (f"{CRL}_forecast_noise_80", "forecast_noise"),
    (f"{CRL}_forecast_origins_noise", "forecast_origins_noise"),
    ("fig16_beta_recon_noise_infection_data_only", "beta_recon_noise"),
    (f"{CRL}_coverage_by_horizon_aci", "conformal_coverage_by_horizon_aci"),
    (f"{CRL}_coverage_by_horizon_split", "conformal_coverage_by_horizon_split"),
]


def collect(figures, folder):
    """Copy figures into folder, numbered in list order. The folder is rebuilt from scratch so
    renumbering never leaves old files behind."""
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    for k, (name, new) in enumerate(figures, start=1):
        new = f"{k:02d}_{new}"
        for ext in ("pdf", "png"):
            f = os.path.join(src, f"{name}.{ext}")
            if os.path.exists(f):
                shutil.copy(f, os.path.join(folder, f"{new}.{ext}"))
            elif ext == "pdf":
                raise FileNotFoundError(f)
    print(f"copied {len(figures)} figures to {folder}")


collect(SYNTHETIC_DATA, os.path.join(dst, "1_synthetic_data"))
collect(PARAMS_PROVIDED, os.path.join(dst, "2_synthetic_params_provided"))
shutil.copy(f"{RES}/rolling/conformal/rolling_conformal_coverage.csv", os.path.join(dst, "2_synthetic_params_provided"))
collect(INFECTION_DATA_ONLY, os.path.join(dst, "3_synthetic_infection_data_only"))
shutil.copy(f"{RES}/rolling_learned/conformal/rolling_conformal_coverage.csv",
            os.path.join(dst, "3_synthetic_infection_data_only"))
shutil.copy(os.path.join(src, "R_recovery", "tables", "R_vs_beta_recovery_summary.csv"),
            os.path.join(dst, "3_synthetic_infection_data_only"))
tables = os.path.join(dst, "tables")
if os.path.isdir(tables):
    shutil.rmtree(tables)
os.makedirs(tables)
for f in os.listdir(os.path.join(src, "tables")):
    if f.endswith(".csv") and f != "beta_recon_infection_data_only_constant_beta.csv":   ### stale single-seed table
        shutil.copy(os.path.join(src, "tables", f), os.path.join(tables, f))
