# app_v2.py  — Strategy 2 dashboard (one-step recursive)
import os
import streamlit as st
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset

st.set_page_config(page_title="Strategy 2 — Demand Forecasts (3M Ahead)", page_icon="📈", layout="wide")

# ========= Config (Strategy 2 only) =========
BASE = "Globle Model/artifacts/v2_recursive"   # folder created in Steps 6–14
MONTHLY_PATH = "Monthly Merge/monthly_data.csv" # raw/clean monthly CSV with YearMonth, Material, demand

# ========= Utilities =========
@st.cache_data
def load_csv(path, parse_dates=None):
    try:
        df = pd.read_csv(path)
        if parse_dates:
            for c in parse_dates:
                if c in df.columns:
                    df[c] = pd.to_datetime(df[c], errors="coerce")
        return df
    except Exception as e:
        st.warning(f"Could not read {path}: {e}")
        return pd.DataFrame()

def ensure_actual_ton(monthly_df: pd.DataFrame) -> pd.DataFrame:
    df = monthly_df.copy()
    if "Monthly_Demand_ton" in df.columns:
        df["Actual_Ton"] = pd.to_numeric(df["Monthly_Demand_ton"], errors="coerce").abs()
    elif "Monthly_Demand" in df.columns:
        df["Actual_Ton"] = pd.to_numeric(df["Monthly_Demand"], errors="coerce").abs() / 1000.0
    elif "Demand_Ton" in df.columns:
        df["Actual_Ton"] = pd.to_numeric(df["Demand_Ton"], errors="coerce")
    else:
        st.error("monthly_data.csv must contain Monthly_Demand_ton, Monthly_Demand, or Demand_Ton.")
        st.stop()
    return df

def ensure_y_true(df: pd.DataFrame, monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Attach y_true to test rows by (Material, TargetMonth) merge if missing."""
    if df.empty or "y_true" in df.columns:
        return df
    if "Material" not in df.columns or "TargetMonth" not in df.columns:
        return df
    df = df.copy()
    df["TargetMonth"] = pd.to_datetime(df["TargetMonth"], errors="coerce")
    join_src = (monthly_df[["Material","YearMonth","Actual_Ton"]]
                .rename(columns={"YearMonth":"TargetMonth","Actual_Ton":"y_true"}))
    df = df.merge(join_src, on=["Material","TargetMonth"], how="left")
    return df

def evaluate_forecast(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if y_true.size == 0:
        return {"MAE%": np.nan, "Bias%": np.nan, "WAPE": np.nan, "RMSE": np.nan, "Score": np.nan}
    mae  = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    y_mean = np.mean(y_true); y_sum = np.sum(y_true)
    mae_pct  = (mae / y_mean) * 100 if y_mean != 0 else np.nan
    bias_pct = (np.mean(y_pred - y_true) / y_mean) * 100 if y_mean != 0 else np.nan  # + overforecast
    wape     = (np.sum(np.abs(y_true - y_pred)) / y_sum) * 100 if y_sum != 0 else np.nan
    score    = mae_pct + np.abs(bias_pct) if np.isfinite(mae_pct) and np.isfinite(bias_pct) else np.nan
    return {"MAE%": mae_pct, "Bias%": bias_pct, "WAPE": wape, "RMSE": rmse, "Score": score}

def fmt_pct(x):
    return "" if pd.isna(x) else f"{x:.2f}%"

# ========= Load data (Strategy 2 artifacts) =========
monthly_data = load_csv(MONTHLY_PATH, parse_dates=["YearMonth"])
monthly_data = ensure_actual_ton(monthly_data)

metrics_ml_overall   = load_csv(os.path.join(BASE, "metrics_ml_overall.csv"))
metrics_ens_overall  = load_csv(os.path.join(BASE, "metrics_ensemble_overall.csv"))
metrics_base_overall = load_csv(os.path.join(BASE, "metrics_baselines_overall.csv"))
metrics_ml_per_sku   = load_csv(os.path.join(BASE, "metrics_ml_per_sku.csv"))

test_ml    = load_csv(os.path.join(BASE, "test_forecasts_ml.csv"), parse_dates=["YearMonth","TargetMonth"])
test_ens   = load_csv(os.path.join(BASE, "test_forecasts_ensemble.csv"), parse_dates=["YearMonth","TargetMonth"])
test_base  = load_csv(os.path.join(BASE, "test_forecasts_baselines.csv"), parse_dates=["YearMonth","TargetMonth"])
train_fit  = load_csv(os.path.join(BASE, "train_fits_ml.csv"), parse_dates=["YearMonth","TargetMonth"])

future_3m  = load_csv(os.path.join(BASE, "future_forecasts_3m.csv"), parse_dates=["BaseMonth","ForecastMonth"])
champion   = load_csv(os.path.join(BASE, "champion_selection.csv"))

# Normalize
for df in [metrics_ml_overall, metrics_ens_overall, metrics_base_overall, metrics_ml_per_sku, test_ml, test_ens, test_base, train_fit]:
    if not df.empty and "Horizon" in df.columns:
        df["Horizon"] = pd.to_numeric(df["Horizon"], errors="coerce")

# Ensure y_true on tests
test_ml  = ensure_y_true(test_ml, monthly_data)
test_ens = ensure_y_true(test_ens, monthly_data)
test_base= ensure_y_true(test_base, monthly_data)

# ========= Sidebar =========
st.sidebar.header("Controls")
sku_list = sorted(monthly_data["Material"].unique().tolist())
sku = st.sidebar.selectbox("Select SKU (Material)", sku_list)
# One-step strategy uses Horizon=1 for model fit; keep chart horizon selector for consistency if needed later.
# horizon = st.sidebar.selectbox("Fit horizon (display)", [1], index=0)

# Champion info
champion_map = {}
if not champion.empty and "Horizon" in champion.columns:
    for _, r in champion.iterrows():
        try:
            champion_map[int(r.get("Horizon", 1))] = {
                "model": str(r.get("model") or r.get("Model") or ""),
                "w": float(r.get("w", 1.0))
            }
        except Exception:
            continue

# ========= Header KPIs =========
st.title("📈 Strategy 2 — 3-Month Forecasts")

colA, colB, colC = st.columns(3)
with colA:
    st.metric("SKUs", f"{len(sku_list)}")
with colB:
    last_date = monthly_data["YearMonth"].max()
    st.metric("Last Actual Month", str(pd.to_datetime(last_date).date()) if pd.notna(last_date) else "-")
with colC:
    if champion_map:
        champ = champion_map.get(1, {})
        st.metric("Champion (one-step)", f"{champ.get('model','-')}")

st.divider()

# ========= Per-SKU section =========
st.subheader(f"SKU: {sku}")

# Actuals
actual = (monthly_data[monthly_data["Material"] == sku]
          .sort_values("YearMonth")[["YearMonth", "Actual_Ton"]]
          .rename(columns={"YearMonth": "Date", "Actual_Ton": "Actual"}))

# Train model fit (LightGBM) aligned to TargetMonth (Horizon=1)
fit_df = train_fit.query("Material == @sku and Horizon == 1 and Model == 'LightGBM'").copy()
if not fit_df.empty:
    fit_df = fit_df.rename(columns={"TargetMonth":"Date", "target":"Train_True", "y_pred":"Fit"})
    fit_df = fit_df[["Date","Fit"]].sort_values("Date")
else:
    fit_df = pd.DataFrame(columns=["Date","Fit"])

# Future forecasts (3 months)
fut = future_3m[future_3m["Material"] == sku].copy()
if not fut.empty:
    fut["Series"] = fut["Horizon"].map(lambda h: f"Forecast (h{int(h)})")
    fut_table = fut.sort_values(["ForecastMonth", "Horizon"])[
        ["ForecastMonth","Horizon","Forecast","ChampionModel","Ensemble_w"]
    ].rename(columns={"ForecastMonth":"Month"})
    total_3m = float(fut["Forecast"].sum())
else:
    fut_table = pd.DataFrame(columns=["Month","Horizon","Forecast","ChampionModel","Ensemble_w"])
    total_3m = np.nan

# ========= Metrics (mean across test months for this SKU) =========
def sku_mean_metrics(rows_df, model_name):
    df = rows_df.query("Material == @sku").copy()
    if df.empty:
        return {"Model": model_name, "MAE%": np.nan, "Bias%": np.nan, "Score": np.nan}
    if "y_true" not in df.columns:
        df = ensure_y_true(df, monthly_data)
    m = evaluate_forecast(df["y_true"].values, df["y_pred"].values)
    return {"Model": model_name, "MAE%": m["MAE%"], "Bias%": m["Bias%"], "Score": m["Score"]}

st.markdown("### Metrics (mean over test months for this SKU)")
met_rows = []
if not test_ml.empty:
    met_rows.append(sku_mean_metrics(test_ml, "LightGBM"))
if not test_ens.empty:
    met_rows.append(sku_mean_metrics(test_ens, "Ensemble"))
# (Optional) show best baseline too
if not test_base.empty:
    met_rows.append(sku_mean_metrics(test_base.query("Model=='MA3'"), "MA3"))
metrics_tbl = pd.DataFrame(met_rows)
if not metrics_tbl.empty:
    disp = metrics_tbl.copy()
    for c in ["MAE%","Bias%","Score"]:
        disp[c] = disp[c].map(fmt_pct)
    st.dataframe(disp.set_index("Model"), use_container_width=True)
else:
    st.info("No test rows found to compute metrics for this SKU.")

# Total next 3 months KPI
st.metric("Total next 3 months (ton)", "-" if pd.isna(total_3m) else f"{total_3m:.2f}")

# ========= Next 3 months table =========
st.markdown("### Next 3 Months (All Horizons)")
if not fut_table.empty:
    st.dataframe(
        fut_table.rename(columns={"Forecast":"Forecast (ton)", "ChampionModel":"Champion Model", "Ensemble_w":"Ensemble w"}),
        hide_index=True, use_container_width=True
    )
else:
    st.info("No future forecast rows found for this SKU.")

# ========= Timeline (History, Fit, Future) =========
st.markdown("### Timeline (History, Fit, Future)")
chart_df = []
if not actual.empty:
    a = actual.copy(); a["Series"] = "Actual"; a = a.rename(columns={"Actual":"Value"})
    chart_df.append(a[["Date","Series","Value"]])
if not fit_df.empty:
    f = fit_df.copy(); f["Series"] = "Model Fit (h1)"; f = f.rename(columns={"Fit":"Value"})
    chart_df.append(f[["Date","Series","Value"]])
if not fut.empty:
    fut_plot = fut[["ForecastMonth","Horizon","Forecast"]].rename(columns={"ForecastMonth":"Date","Forecast":"Value"})
    fut_plot["Series"] = fut_plot["Horizon"].map(lambda h: f"Forecast (h{int(h)})")
    chart_df.append(fut_plot[["Date","Series","Value"]])

chart_df = pd.concat(chart_df, ignore_index=True) if chart_df else pd.DataFrame()
if chart_df.empty:
    st.info("Nothing to plot for this SKU yet.")
else:
    chart_df = chart_df.sort_values("Date")
    import altair as alt
    line = alt.Chart(chart_df).mark_line(point=True).encode(
        x=alt.X("Date:T", title="Month"),
        y=alt.Y("Value:Q", title="Demand (ton)"),
        color=alt.Color("Series:N", title=""),
        tooltip=[alt.Tooltip("Date:T", title="Month"),
                 alt.Tooltip("Series:N"),
                 alt.Tooltip("Value:Q", title="Value", format=".2f")]
    ).properties(height=420)
    st.altair_chart(line, use_container_width=True)

# ========= Test Forecast vs Actual (Holdout) =========
st.markdown("### Test Forecast vs Actual (Holdout)")
test_plot_parts = []

# Only LightGBM rows from ML test prediction
if not test_ml.empty:
    if "Model" in test_ml.columns:
        ml_rows = test_ml.query("Material == @sku and Horizon == 1 and Model == 'LightGBM'").copy()
    else:
        # Fallback if Model column not present (shouldn't happen)
        ml_rows = test_ml.query("Material == @sku and Horizon == 1").copy()

    ml_rows = ensure_y_true(ml_rows, monthly_data)
    if not ml_rows.empty:
        ml_rows = ml_rows.rename(columns={"TargetMonth": "Date"})
        pred_lgb = ml_rows[["Date", "y_pred"]].assign(Series="Test Pred (LightGBM)").rename(columns={"y_pred": "Value"})
        act      = ml_rows[["Date", "y_true"]].assign(Series="Actual (Holdout)").rename(columns={"y_true": "Value"})
        test_plot_parts += [pred_lgb, act]

# Ensemble test rows (already only ensemble)
if not test_ens.empty:
    ens_rows = test_ens.query("Material == @sku and Horizon == 1").copy()
    ens_rows = ensure_y_true(ens_rows, monthly_data)
    if not ens_rows.empty:
        ens_rows = ens_rows.rename(columns={"TargetMonth": "Date"})
        pred_ens = ens_rows[["Date", "y_pred"]].assign(Series="Test Pred (Ensemble)").rename(columns={"y_pred": "Value"})
        test_plot_parts.append(pred_ens)

test_plot_df = pd.concat(test_plot_parts, ignore_index=True) if test_plot_parts else pd.DataFrame()
if test_plot_df.empty:
    st.info("No test-set predictions found for this SKU.")
else:
    import altair as alt
    test_plot_df = test_plot_df.sort_values("Date")
    layer_actual = alt.Chart(test_plot_df[test_plot_df["Series"] == "Actual (Holdout)"]).mark_point(size=80).encode(
        x=alt.X("Date:T", title="Target Month"),
        y=alt.Y("Value:Q", title="Demand (ton)"),
        color=alt.Color("Series:N", title="")
    )
    layer_lgb = alt.Chart(test_plot_df[test_plot_df["Series"] == "Test Pred (LightGBM)"]).mark_line(point=True).encode(
        x="Date:T", y="Value:Q", color=alt.Color("Series:N", title="")
    )
    layer_ens = alt.Chart(test_plot_df[test_plot_df["Series"] == "Test Pred (Ensemble)"]).mark_line(point=True).encode(
        x="Date:T", y="Value:Q", color=alt.Color("Series:N", title="")
    )
    st.altair_chart((layer_actual + layer_lgb + layer_ens).properties(height=380), use_container_width=True)


# ========= Downloads =========
st.markdown("### Downloads")
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    if not metrics_ml_overall.empty:
        st.download_button("⬇️ ML Overall", data=metrics_ml_overall.to_csv(index=False), file_name="metrics_ml_overall.csv")
with c2:
    if not metrics_ens_overall.empty:
        st.download_button("⬇️ Ensemble Overall", data=metrics_ens_overall.to_csv(index=False), file_name="metrics_ensemble_overall.csv")
with c3:
    if not test_ml.empty:
        st.download_button("⬇️ Test Forecasts (ML)", data=test_ml.to_csv(index=False), file_name="test_forecasts_ml.csv")
with c4:
    if not test_ens.empty:
        st.download_button("⬇️ Test Forecasts (Ensemble)", data=test_ens.to_csv(index=False), file_name="test_forecasts_ensemble.csv")
with c5:
    if not future_3m.empty:
        st.download_button("⬇️ Future Forecasts (3M)", data=future_3m.to_csv(index=False), file_name="future_forecasts_3m.csv")

st.caption("Strategy 2: one-step model with recursive 3-month roll-forward. "
           "Metrics (MAE%, Bias%, Score) computed on test months for the selected SKU. "
           "Total next 3 months sums M+1..M+3 forecasts.")
