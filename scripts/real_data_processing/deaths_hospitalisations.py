import pandas as pd
import numpy as np

N = 56_000_000

ADMISSIONS_CSV = "../../Data/england_admissions_daily.csv"
DEATHS_CSV = "../../Data/england_deaths_ONSByDay.csv"

DATE_COL  = "date"
VALUE_COL = "metric_value"     # both your files use this name

def load_daily_fraction(csv_path, label):
    df = pd.read_csv(csv_path)

    ### filter to England only if an area column exists
    if "geography" in df.columns:
        df = df[df["geography"] == "England"]

    for col, keep in [("sex", "all"), ("age", "all"), ("stratum", "default")]:
        if col in df.columns and keep in set(df[col].astype(str)):
            df = df[df[col].astype(str) == keep]

    df = df[[DATE_COL, VALUE_COL]].rename(columns={VALUE_COL: "raw"})
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], dayfirst=True)  
    df = df.sort_values(DATE_COL)
    df = df.groupby(DATE_COL, as_index=False)["raw"].sum()     
    df.loc[df["raw"] < 0, "raw"] = 0.0                     

    ### 7 day rolling average
    df["smooth"] = df["raw"].rolling(7, center=True, min_periods=1).mean()
    s = pd.Series((df["smooth"] / N).values, index=df[DATE_COL])
    print(f"  {label:10s}: {len(s)} daily rows "
          f"({s.index.min().date()} -> {s.index.max().date()})")
    return s


def to_cases_grid(daily, cases_dates):
    cases_dates = pd.to_datetime(cases_dates)
    out = np.full(len(cases_dates), np.nan)
    for i, d0 in enumerate(cases_dates):
        d1 = d0 + pd.Timedelta(days=7)
        window = daily[(daily.index >= d0) & (daily.index < d1)]
        if len(window):
            out[i] = window.mean()
    return np.nan_to_num(out, nan=0.0).reshape(-1, 1)


def main():
    cases_dates = np.load("../../data/dates_study.npy").astype("datetime64[D]")
    I = np.load("../../data/I_data_study.npy").reshape(-1, 1)
    cases_dates = cases_dates[:len(I)]
    print(f"cases grid: {len(cases_dates)} weeks "
          f"({pd.Timestamp(cases_dates[0]).date()} -> {pd.Timestamp(cases_dates[-1]).date()})")

    print("loading daily series:")
    h_daily = load_daily_fraction(ADMISSIONS_CSV, "admissions")
    d_daily = load_daily_fraction(DEATHS_CSV,     "deaths")

    H = to_cases_grid(h_daily, cases_dates)
    D = to_cases_grid(d_daily, cases_dates)

    np.save("../../data/H_data_study.npy", H)
    np.save("../../data/D_data_study.npy", D)

    print(f"\nsaved H_data_study.npy  len {len(H)}  min {H.min():.3e}  max {H.max():.3e}")
    print(f"saved D_data_study.npy  len {len(D)}  min {D.min():.3e}  max {D.max():.3e}")
    n_zero_H = int((H.reshape(-1) == 0).sum())
    n_zero_D = int((D.reshape(-1) == 0).sum())
    if not (len(I) == len(H) == len(D)):
        print("WARNING: lengths differ from cases -- alignment broken.")
    else:
        print(f"OK: all three series length {len(I)}.")
    if n_zero_H or n_zero_D:
        print(f"note: {n_zero_H} zero weeks in admissions, {n_zero_D} in deaths "
              f"(weeks with no overlapping daily data -> filled 0).")


if __name__ == "__main__":
    main()