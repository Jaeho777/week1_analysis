from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from darts import TimeSeries
from statsmodels.tsa.seasonal import STL


ADI_THRESHOLD = 1.32
CV2_THRESHOLD = 0.49
MIN_EFFECTIVE_WEEKS = 24
VALID_CUM_QTY_PCT = 0.05


@dataclass
class Paths:
    input_dir: Path
    output_dir: Path
    tables_dir: Path
    figures_dir: Path
    reports_dir: Path


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_input = repo_root.parent / "mcm"
    default_output = script_dir / "output"
    parser = argparse.ArgumentParser(
        description="Build a fresh Darts-based MCM EDA and decision pipeline."
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--forecast-horizon", type=int, default=12)
    parser.add_argument("--top-sku-count", type=int, default=6)
    return parser.parse_args()


def init_paths(input_dir: Path, output_dir: Path) -> Paths:
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"
    for path in [output_dir, tables_dir, figures_dir, reports_dir]:
        path.mkdir(parents=True, exist_ok=True)
    for folder, patterns in [
        (tables_dir, ["*.csv"]),
        (figures_dir, ["*.png"]),
        (reports_dir, ["*.md"]),
    ]:
        for pattern in patterns:
            for old_file in folder.glob(pattern):
                old_file.unlink()
    return Paths(input_dir, output_dir, tables_dir, figures_dir, reports_dir)


def read_parquet(input_dir: Path, stem: str, columns: list[str] | None = None) -> pd.DataFrame:
    matches = sorted(
        p
        for p in input_dir.glob(f"{stem}_*.parquet")
        if p.name.startswith(f"{stem}_")
        and len(p.name) > len(stem) + 1
        and p.name[len(stem) + 1].isdigit()
    )
    if not matches:
        raise FileNotFoundError(f"No parquet file found for {stem!r} in {input_dir}")
    return pq.read_table(matches[-1], columns=columns).to_pandas()


def parse_yyyymmdd(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.isin(["", "00000000", "99991231"]))
    return pd.to_datetime(cleaned, format="%Y%m%d", errors="coerce")


def save_table(df: pd.DataFrame, paths: Paths, name: str) -> Path:
    path = paths.tables_dir / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_figure(fig: plt.Figure, paths: Paths, name: str) -> Path:
    path = paths.figures_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def add_value_labels(ax: plt.Axes, fmt: str = "{:,.0f}") -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if not np.isfinite(height) or height == 0:
            continue
        ax.annotate(
            fmt.format(height),
            (patch.get_x() + patch.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 2),
            textcoords="offset points",
        )


def demand_class(adi: float, cv2: float) -> str:
    if not np.isfinite(adi) or not np.isfinite(cv2):
        return "Insufficient"
    if adi < ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        return "Smooth"
    if adi < ADI_THRESHOLD and cv2 >= CV2_THRESHOLD:
        return "Erratic"
    if adi >= ADI_THRESHOLD and cv2 < CV2_THRESHOLD:
        return "Intermittent"
    return "Lumpy"


def assign_abc(stats: pd.DataFrame, metric: str = "total_qty") -> pd.DataFrame:
    ranked = stats.sort_values(metric, ascending=False).copy()
    total = ranked[metric].sum()
    ranked["cum_share"] = ranked[metric].cumsum() / total if total else 0
    ranked["abc_class"] = np.select(
        [ranked["cum_share"] <= 0.80, ranked["cum_share"] <= 0.95],
        ["A", "B"],
        default="C",
    )
    return ranked[["sku_cd", "abc_class", "cum_share"]]


def build_dataset_overview(paths: Paths) -> pd.DataFrame:
    rows = []
    for parquet_path in sorted(paths.input_dir.glob("*.parquet")):
        pf = pq.ParquetFile(parquet_path)
        rows.append(
            {
                "dataset": parquet_path.name,
                "rows": pf.metadata.num_rows,
                "columns": pf.metadata.num_columns,
                "size_mb": parquet_path.stat().st_size / 1024 / 1024,
                "row_groups": pf.metadata.num_row_groups,
            }
        )
    overview = pd.DataFrame(rows)
    save_table(overview, paths, "01_dataset_overview")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    plot_df = overview.sort_values("rows", ascending=False)
    sns.barplot(data=plot_df, x="rows", y="dataset", ax=ax, color="#4974a5")
    ax.set_title("Source parquet row counts")
    ax.set_xlabel("Rows")
    ax.set_ylabel("")
    save_figure(fig, paths, "01_dataset_overview_rows")
    return overview


def build_sales_base(paths: Paths) -> tuple[pd.DataFrame, pd.DataFrame]:
    sales = read_parquet(
        paths.input_dir,
        "sales",
        [
            "company_cd",
            "region",
            "receipt_no",
            "store_cd",
            "sales_date",
            "sku_cd",
            "sales_qty",
            "sales_amt",
            "net_amt",
            "dc_amt",
        ],
    )
    sales["sales_date"] = parse_yyyymmdd(sales["sales_date"])
    sales["sales_qty"] = pd.to_numeric(sales["sales_qty"], errors="coerce").fillna(0.0)
    sales["sales_amt"] = pd.to_numeric(sales["sales_amt"], errors="coerce").fillna(0.0)
    sales["net_amt"] = pd.to_numeric(sales["net_amt"], errors="coerce").fillna(0.0)
    sales["dc_amt"] = pd.to_numeric(sales["dc_amt"], errors="coerce").fillna(0.0)
    sales["positive_qty"] = sales["sales_qty"].clip(lower=0.0)
    sales["return_qty"] = (-sales["sales_qty"].clip(upper=0.0)).fillna(0.0)
    sales = sales.dropna(subset=["sales_date", "sku_cd"])

    daily_sku = (
        sales.groupby(["sku_cd", "sales_date"], as_index=False, observed=True)
        .agg(
            qty=("positive_qty", "sum"),
            gross_qty=("sales_qty", "sum"),
            sales_amt=("sales_amt", "sum"),
            net_amt=("net_amt", "sum"),
            dc_amt=("dc_amt", "sum"),
            receipt_count=("receipt_no", "nunique"),
        )
        .sort_values(["sku_cd", "sales_date"])
    )
    save_table(
        pd.DataFrame(
            [
                {
                    "sales_rows": len(sales),
                    "daily_sku_rows": len(daily_sku),
                    "sku_count": sales["sku_cd"].nunique(),
                    "store_count": sales["store_cd"].nunique(),
                    "date_min": sales["sales_date"].min().date().isoformat(),
                    "date_max": sales["sales_date"].max().date().isoformat(),
                    "total_positive_qty": sales["positive_qty"].sum(),
                    "total_net_amt": sales["net_amt"].sum(),
                }
            ]
        ),
        paths,
        "02_sales_base_summary",
    )
    return sales, daily_sku


def validate_effective_history(paths: Paths, daily_sku: pd.DataFrame) -> pd.DataFrame:
    global_max_date = daily_sku["sales_date"].max()
    positive = daily_sku[daily_sku["qty"] > 0].copy()
    if positive.empty:
        raise ValueError("No positive sales quantity found.")

    total_qty = positive.groupby("sku_cd", as_index=False)["qty"].sum().rename(
        columns={"qty": "total_qty"}
    )
    active_days = positive.groupby("sku_cd", as_index=False)["sales_date"].nunique().rename(
        columns={"sales_date": "active_days"}
    )
    positive = positive.merge(total_qty, on="sku_cd", how="left")
    positive["cum_qty"] = positive.groupby("sku_cd")["qty"].cumsum()
    threshold_rows = positive[
        positive["cum_qty"] >= positive["total_qty"] * VALID_CUM_QTY_PCT
    ]
    first_valid = (
        threshold_rows.groupby("sku_cd", as_index=False)["sales_date"]
        .first()
        .rename(columns={"sales_date": "first_valid_date"})
    )
    first_last = (
        positive.groupby("sku_cd", as_index=False)
        .agg(first_sale_date=("sales_date", "min"), last_sale_date=("sales_date", "max"))
    )
    sku_stats = total_qty.merge(active_days, on="sku_cd", how="outer")
    sku_stats = sku_stats.merge(first_last, on="sku_cd", how="left")
    sku_stats = sku_stats.merge(first_valid, on="sku_cd", how="left")
    sku_stats["effective_days"] = (
        global_max_date - sku_stats["first_valid_date"]
    ).dt.days + 1
    sku_stats["effective_weeks"] = sku_stats["effective_days"] / 7
    sku_stats["eligible_min_24_weeks"] = sku_stats["effective_weeks"] >= MIN_EFFECTIVE_WEEKS
    sku_stats["filter_reason"] = np.select(
        [
            sku_stats["total_qty"].fillna(0) <= 0,
            ~sku_stats["eligible_min_24_weeks"].fillna(False),
        ],
        ["no_positive_demand", "effective_history_less_than_24_weeks"],
        default="pass",
    )

    all_skus = pd.DataFrame({"sku_cd": daily_sku["sku_cd"].drop_duplicates()})
    sku_stats = all_skus.merge(sku_stats, on="sku_cd", how="left")
    sku_stats["filter_reason"] = sku_stats["filter_reason"].fillna("no_positive_demand")
    sku_stats["eligible_min_24_weeks"] = sku_stats["eligible_min_24_weeks"].fillna(False)
    save_table(sku_stats, paths, "03_valid_history_by_sku")
    summary = (
        sku_stats.groupby(["eligible_min_24_weeks", "filter_reason"], as_index=False)
        .agg(sku_count=("sku_cd", "nunique"), avg_effective_weeks=("effective_weeks", "mean"))
        .sort_values(["eligible_min_24_weeks", "filter_reason"])
    )
    save_table(summary, paths, "03_valid_history_summary")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(
        data=sku_stats,
        x="effective_weeks",
        bins=40,
        ax=axes[0],
        color="#4f8f7b",
    )
    axes[0].axvline(MIN_EFFECTIVE_WEEKS, color="#b64242", linestyle="--", label="24 weeks")
    axes[0].set_title("Effective demand history length")
    axes[0].set_xlabel("Effective weeks")
    axes[0].legend()
    sns.countplot(data=sku_stats, y="filter_reason", ax=axes[1], color="#6f7fb5")
    axes[1].set_title("Validation filter outcome")
    axes[1].set_xlabel("SKU count")
    axes[1].set_ylabel("")
    save_figure(fig, paths, "03_validation_effective_history")
    return sku_stats


def profile_demand(paths: Paths, daily_sku: pd.DataFrame, valid_stats: pd.DataFrame) -> pd.DataFrame:
    analysis = valid_stats[
        ["sku_cd", "first_valid_date", "effective_days", "effective_weeks", "eligible_min_24_weeks"]
    ].copy()
    positive = daily_sku[daily_sku["qty"] > 0].merge(
        analysis[["sku_cd", "first_valid_date"]], on="sku_cd", how="left"
    )
    positive = positive[positive["sales_date"] >= positive["first_valid_date"]]

    grouped = (
        positive.groupby("sku_cd", as_index=False)
        .agg(
            total_qty=("qty", "sum"),
            active_days=("sales_date", "nunique"),
            avg_positive_qty=("qty", "mean"),
            std_positive_qty=("qty", "std"),
            total_sales_amt=("sales_amt", "sum"),
            total_net_amt=("net_amt", "sum"),
        )
        .merge(analysis, on="sku_cd", how="right")
    )
    grouped["total_qty"] = grouped["total_qty"].fillna(0.0)
    grouped["active_days"] = grouped["active_days"].fillna(0.0)
    grouped["std_positive_qty"] = grouped["std_positive_qty"].fillna(0.0)
    grouped["avg_positive_qty"] = grouped["avg_positive_qty"].replace(0, np.nan)
    grouped["adi"] = grouped["effective_days"] / grouped["active_days"].replace(0, np.nan)
    grouped["cv2"] = (grouped["std_positive_qty"] / grouped["avg_positive_qty"]) ** 2
    grouped["cv2"] = grouped["cv2"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    grouped["zero_ratio"] = 1 - (
        grouped["active_days"] / grouped["effective_days"].replace(0, np.nan)
    )
    grouped["demand_class"] = [demand_class(a, c) for a, c in zip(grouped["adi"], grouped["cv2"])]

    abc = assign_abc(grouped[grouped["total_qty"] > 0], "total_qty")
    grouped = grouped.merge(abc, on="sku_cd", how="left")
    grouped["abc_class"] = grouped["abc_class"].fillna("No Sales")
    save_table(grouped, paths, "04_demand_profile_by_sku")

    class_summary = (
        grouped.groupby("demand_class", as_index=False)
        .agg(
            sku_count=("sku_cd", "nunique"),
            total_qty=("total_qty", "sum"),
            avg_adi=("adi", "mean"),
            avg_cv2=("cv2", "mean"),
            avg_zero_ratio=("zero_ratio", "mean"),
        )
        .sort_values("total_qty", ascending=False)
    )
    save_table(class_summary, paths, "04_demand_class_summary")

    abc_summary = (
        grouped.groupby("abc_class", as_index=False)
        .agg(sku_count=("sku_cd", "nunique"), total_qty=("total_qty", "sum"))
        .sort_values("abc_class")
    )
    save_table(abc_summary, paths, "04_abc_summary")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    scatter_df = grouped[(grouped["total_qty"] > 0) & grouped["adi"].notna()].copy()
    scatter_df["plot_adi"] = scatter_df["adi"].clip(upper=scatter_df["adi"].quantile(0.99))
    scatter_df["plot_cv2"] = scatter_df["cv2"].clip(upper=scatter_df["cv2"].quantile(0.99))
    sns.scatterplot(
        data=scatter_df,
        x="plot_adi",
        y="plot_cv2",
        hue="demand_class",
        size="total_qty",
        sizes=(8, 80),
        alpha=0.55,
        linewidth=0,
        ax=axes[0],
    )
    axes[0].axvline(ADI_THRESHOLD, color="#222222", linestyle="--", linewidth=1)
    axes[0].axhline(CV2_THRESHOLD, color="#222222", linestyle="--", linewidth=1)
    axes[0].set_title("ADI / CV2 demand classification")
    axes[0].set_xlabel("ADI, clipped at p99")
    axes[0].set_ylabel("CV2, clipped at p99")
    axes[0].legend(fontsize=7, loc="upper right")
    sns.barplot(data=class_summary, x="demand_class", y="total_qty", ax=axes[1], color="#c47a45")
    axes[1].set_title("Demand quantity by class")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Total quantity")
    axes[1].tick_params(axis="x", rotation=25)
    add_value_labels(axes[1])
    save_figure(fig, paths, "04_demand_profile_adi_cv2")

    pareto = grouped[grouped["total_qty"] > 0].sort_values("total_qty", ascending=False).copy()
    pareto["sku_rank"] = np.arange(1, len(pareto) + 1)
    pareto["cum_qty_share"] = pareto["total_qty"].cumsum() / pareto["total_qty"].sum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(pareto["sku_rank"], pareto["cum_qty_share"], color="#315e8a", linewidth=2)
    ax.axhline(0.8, color="#b64242", linestyle="--", label="80% quantity")
    ax.set_title("SKU demand Pareto curve")
    ax.set_xlabel("SKU rank by quantity")
    ax.set_ylabel("Cumulative quantity share")
    ax.legend()
    save_figure(fig, paths, "04_abc_pareto_curve")
    return grouped


def make_weekly_timeseries(daily: pd.DataFrame, date_col: str, value_col: str) -> TimeSeries:
    weekly = (
        daily.set_index(date_col)[value_col]
        .sort_index()
        .resample("W-MON")
        .sum()
        .rename(value_col)
        .reset_index()
    )
    return TimeSeries.from_dataframe(weekly, time_col=date_col, value_cols=value_col, freq="W-MON")


def structural_darts_analysis(
    paths: Paths,
    daily_sku: pd.DataFrame,
    demand_profile: pd.DataFrame,
    forecast_horizon: int,
    top_sku_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_total = (
        daily_sku.groupby("sales_date", as_index=False)["qty"].sum().sort_values("sales_date")
    )
    overall_series = make_weekly_timeseries(daily_total, "sales_date", "qty")
    overall_weekly = overall_series.to_dataframe().reset_index()
    overall_weekly.columns = ["week", "qty"]
    save_table(overall_weekly, paths, "05_darts_weekly_total_series")

    fig, ax = plt.subplots(figsize=(12, 4.8))
    overall_series.plot(ax=ax, label="Weekly demand")
    ax.set_title("Darts TimeSeries weekly total demand")
    ax.set_xlabel("Week")
    ax.set_ylabel("Quantity")
    ax.legend()
    save_figure(fig, paths, "05_darts_weekly_total_series")

    stl_df = overall_weekly.dropna().copy()
    period = 52 if len(stl_df) >= 104 else max(2, min(26, len(stl_df) // 2))
    stl = STL(stl_df["qty"].astype(float), period=period, robust=True).fit()
    components = pd.DataFrame(
        {
            "week": stl_df["week"],
            "observed": stl_df["qty"].to_numpy(),
            "trend": stl.trend,
            "seasonal": stl.seasonal,
            "residual": stl.resid,
        }
    )
    components["residual_z"] = (
        components["residual"] - components["residual"].mean()
    ) / components["residual"].std(ddof=0)
    components["residual_anomaly"] = components["residual_z"].abs() >= 2.5
    save_table(components, paths, "05_stl_components")

    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    for ax, col, color in zip(
        axes,
        ["observed", "trend", "seasonal", "residual"],
        ["#2f5d7c", "#4f8f7b", "#9a6fb0", "#b6624f"],
    ):
        ax.plot(components["week"], components[col], color=color, linewidth=1.5)
        ax.set_ylabel(col)
    axes[0].set_title("STL decomposition of Darts weekly series")
    save_figure(fig, paths, "05_stl_decomposition")

    profiled_daily = daily_sku.merge(
        demand_profile[["sku_cd", "demand_class", "abc_class"]], on="sku_cd"
    )
    weekly_sku = (
        profiled_daily.groupby(
            ["sku_cd", pd.Grouper(key="sales_date", freq="W-MON"), "demand_class", "abc_class"],
            as_index=False,
        )["qty"]
        .sum()
        .rename(columns={"sales_date": "week"})
    )
    recent_cutoff = weekly_sku["week"].max() - pd.Timedelta(weeks=52)
    recent_rank = (
        weekly_sku[weekly_sku["week"] >= recent_cutoff]
        .groupby("sku_cd", as_index=False)["qty"]
        .sum()
        .rename(columns={"qty": "recent_52w_qty"})
    )
    eligible_recent = (
        demand_profile.query("total_qty > 0 and eligible_min_24_weeks == True")
        .merge(recent_rank, on="sku_cd", how="left")
        .fillna({"recent_52w_qty": 0})
        .sort_values(["recent_52w_qty", "total_qty"], ascending=False)
    )
    top_skus = (
        eligible_recent[eligible_recent["recent_52w_qty"] > 0]
        .head(top_sku_count)["sku_cd"]
        .tolist()
    )
    forecast_rows = []
    fig, axes = plt.subplots(
        math.ceil(max(1, len(top_skus)) / 2),
        2,
        figsize=(13, max(4, 3.4 * math.ceil(max(1, len(top_skus)) / 2))),
        squeeze=False,
    )
    for i, sku in enumerate(top_skus):
        ax = axes[i // 2][i % 2]
        sku_weekly = weekly_sku[weekly_sku["sku_cd"] == sku][["week", "qty"]].sort_values("week")
        full_idx = pd.date_range(sku_weekly["week"].min(), sku_weekly["week"].max(), freq="W-MON")
        sku_weekly = (
            sku_weekly.set_index("week")
            .reindex(full_idx, fill_value=0)
            .rename_axis("week")
            .reset_index()
        )
        series = TimeSeries.from_dataframe(
            sku_weekly, time_col="week", value_cols="qty", fill_missing_dates=True, freq="W-MON"
        )
        history = series.to_dataframe().reset_index()
        history.columns = ["week", "qty"]
        last_week = history["week"].max()
        base = history["qty"].tail(4).mean()
        seasonal = (
            history["qty"].iloc[-52:-52 + forecast_horizon].to_numpy()
            if len(history) >= 52 + forecast_horizon
            else np.repeat(base, forecast_horizon)
        )
        forecast_values = 0.5 * np.repeat(base, forecast_horizon) + 0.5 * seasonal
        forecast_index = pd.date_range(last_week + pd.offsets.Week(weekday=0), periods=forecast_horizon, freq="W-MON")
        forecast_df = pd.DataFrame(
            {
                "sku_cd": sku,
                "week": forecast_index,
                "forecast_qty": np.maximum(0, forecast_values),
                "method": "darts_series_seasonal_moving_average",
            }
        )
        forecast_rows.append(forecast_df)
        ax.plot(history["week"].tail(104), history["qty"].tail(104), label="actual", color="#2f5d7c")
        ax.plot(forecast_df["week"], forecast_df["forecast_qty"], label="forecast", color="#b64242")
        ax.set_title(sku)
        ax.set_xlabel("")
        ax.set_ylabel("Weekly qty")
        ax.legend(fontsize=8)
    for j in range(len(top_skus), axes.size):
        axes[j // 2][j % 2].axis("off")
    save_figure(fig, paths, "05_top_sku_darts_forecasts")

    forecasts = pd.concat(forecast_rows, ignore_index=True) if forecast_rows else pd.DataFrame()
    save_table(forecasts, paths, "05_top_sku_darts_forecasts")

    backtest = []
    candidates = eligible_recent[eligible_recent["recent_52w_qty"] > 0].head(200)["sku_cd"].tolist()
    for sku in candidates:
        sku_weekly = weekly_sku[weekly_sku["sku_cd"] == sku][["week", "qty"]].sort_values("week")
        if len(sku_weekly) < 2:
            continue
        full_idx = pd.date_range(sku_weekly["week"].min(), sku_weekly["week"].max(), freq="W-MON")
        y = sku_weekly.set_index("week")["qty"].reindex(full_idx, fill_value=0).astype(float)
        positive_positions = np.flatnonzero(y.to_numpy() > 0)
        if len(positive_positions) == 0:
            continue
        y = y.iloc[: positive_positions[-1] + 1]
        if len(y) < 12:
            continue
        test_h = min(12, max(4, len(y) // 5))
        if len(y) <= test_h + 4:
            continue
        train = y.iloc[:-test_h]
        actual = y.iloc[-test_h:]
        denom = actual.abs().sum()
        if denom == 0:
            continue
        moving_avg_pred = np.repeat(train.tail(4).mean(), test_h)
        if len(train) >= 52:
            seasonal_seed = train.iloc[-52 : -52 + test_h].to_numpy()
            if len(seasonal_seed) < test_h:
                seasonal_seed = np.resize(seasonal_seed, test_h)
            seasonal_pred = seasonal_seed
        else:
            seasonal_pred = moving_avg_pred
        backtest.append(
            {
                "sku_cd": sku,
                "actual_qty": actual.sum(),
                "ma4_wape": np.abs(actual.to_numpy() - moving_avg_pred).sum() / denom,
                "seasonal_wape": np.abs(actual.to_numpy() - seasonal_pred).sum() / denom,
            }
        )
    backtest_df = pd.DataFrame(backtest)
    save_table(backtest_df, paths, "05_darts_baseline_backtest")
    if not backtest_df.empty:
        melted = backtest_df.melt(
            id_vars=["sku_cd", "actual_qty"],
            value_vars=["ma4_wape", "seasonal_wape"],
            var_name="baseline",
            value_name="wape",
        ).dropna()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(data=melted, x="baseline", y="wape", ax=ax, color="#85a9c5")
        ax.set_title("Baseline forecast WAPE on top 200 SKUs")
        ax.set_xlabel("")
        ax.set_ylabel("WAPE")
        save_figure(fig, paths, "05_darts_baseline_backtest")
    return forecasts, backtest_df


def promotion_stock_segment_analysis(
    paths: Paths,
    sales: pd.DataFrame,
    daily_sku: pd.DataFrame,
    demand_profile: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top_skus = demand_profile.sort_values("total_qty", ascending=False).head(200)["sku_cd"]
    promotion = read_parquet(
        paths.input_dir,
        "promotion",
        ["company_cd", "promo_cd", "promo_type_nm", "promo_nm", "promo_start", "promo_end", "discount_rate"],
    )
    promotion["promo_start_dt"] = parse_yyyymmdd(promotion["promo_start"])
    promotion["promo_end_dt"] = parse_yyyymmdd(promotion["promo_end"])
    promotion["promo_end_dt"] = promotion["promo_end_dt"].fillna(sales["sales_date"].max())
    promotion_item = read_parquet(paths.input_dir, "promotion_item", ["promo_cd", "sku_cd"])
    promo_top = (
        promotion_item[promotion_item["sku_cd"].isin(top_skus)]
        .drop_duplicates()
        .merge(promotion, on="promo_cd", how="left")
        .dropna(subset=["promo_start_dt", "promo_end_dt"])
    )
    promo_top = promo_top[
        (promo_top["promo_end_dt"] >= sales["sales_date"].min())
        & (promo_top["promo_start_dt"] <= sales["sales_date"].max())
    ]

    sales_top = daily_sku[daily_sku["sku_cd"].isin(top_skus)][["sku_cd", "sales_date", "qty"]].copy()
    sales_top["promo_flag"] = False
    for sku, intervals in promo_top.groupby("sku_cd"):
        idx = sales_top["sku_cd"] == sku
        sku_dates = sales_top.loc[idx, "sales_date"]
        flag = pd.Series(False, index=sales_top.loc[idx].index)
        for row in intervals.itertuples(index=False):
            flag |= (sku_dates >= row.promo_start_dt) & (sku_dates <= row.promo_end_dt)
        sales_top.loc[flag.index, "promo_flag"] = flag

    promo_effect = (
        sales_top.groupby(["sku_cd", "promo_flag"], as_index=False)["qty"]
        .mean()
        .pivot(index="sku_cd", columns="promo_flag", values="qty")
        .rename(columns={False: "non_promo_avg_daily_qty", True: "promo_avg_daily_qty"})
        .reset_index()
    )
    for col in ["non_promo_avg_daily_qty", "promo_avg_daily_qty"]:
        if col not in promo_effect.columns:
            promo_effect[col] = np.nan
    promo_effect["promo_lift"] = (
        promo_effect["promo_avg_daily_qty"] / promo_effect["non_promo_avg_daily_qty"].replace(0, np.nan)
        - 1
    )
    promo_effect = promo_effect.merge(
        demand_profile[["sku_cd", "demand_class", "abc_class", "total_qty"]],
        on="sku_cd",
        how="left",
    ).sort_values("promo_lift", ascending=False)
    save_table(promo_effect, paths, "06_promotion_lift_top_skus")

    plot_promo = promo_effect.replace([np.inf, -np.inf], np.nan).dropna(subset=["promo_lift"]).head(20)
    if not plot_promo.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=plot_promo, y="sku_cd", x="promo_lift", hue="demand_class", dodge=False, ax=ax)
        ax.axvline(0, color="#222222", linewidth=1)
        ax.set_title("Promotion lift by top SKU")
        ax.set_xlabel("Avg daily qty lift")
        ax.set_ylabel("")
        ax.legend(fontsize=8)
        save_figure(fig, paths, "06_promotion_lift_top_skus")

    stock = read_parquet(paths.input_dir, "stock", ["sku_cd", "quantity", "aging", "basedate"])
    stock["basedate"] = parse_yyyymmdd(stock["basedate"])
    stock["quantity"] = pd.to_numeric(stock["quantity"], errors="coerce").fillna(0.0)
    stock["aging"] = pd.to_numeric(stock["aging"], errors="coerce")
    latest_date = stock["basedate"].max()
    latest_stock = (
        stock[stock["basedate"] == latest_date]
        .groupby("sku_cd", as_index=False)
        .agg(latest_stock_qty=("quantity", "sum"), avg_aging=("aging", "mean"))
    )
    stock_history = (
        stock.groupby(["sku_cd", "basedate"], as_index=False)["quantity"]
        .sum()
        .assign(stockout=lambda df: df["quantity"] <= 0)
        .groupby("sku_cd", as_index=False)
        .agg(stockout_rate=("stockout", "mean"), stock_snapshots=("basedate", "nunique"))
    )
    sales_max = daily_sku["sales_date"].max()
    recent = daily_sku[daily_sku["sales_date"] >= sales_max - pd.Timedelta(days=84)]
    velocity = (
        recent.groupby("sku_cd", as_index=False)["qty"].sum().rename(columns={"qty": "last_12w_qty"})
    )
    velocity["weekly_velocity"] = velocity["last_12w_qty"] / 12
    stock_risk = (
        demand_profile[["sku_cd", "demand_class", "abc_class", "total_qty"]]
        .merge(latest_stock, on="sku_cd", how="left")
        .merge(stock_history, on="sku_cd", how="left")
        .merge(velocity[["sku_cd", "weekly_velocity"]], on="sku_cd", how="left")
    )
    stock_risk["latest_stock_qty"] = stock_risk["latest_stock_qty"].fillna(0.0)
    stock_risk["weekly_velocity"] = stock_risk["weekly_velocity"].fillna(0.0)
    stock_risk["coverage_weeks"] = stock_risk["latest_stock_qty"] / stock_risk[
        "weekly_velocity"
    ].replace(0, np.nan)
    stock_risk["stock_risk"] = np.select(
        [
            (stock_risk["weekly_velocity"] > 0)
            & ((stock_risk["coverage_weeks"] < 4) | (stock_risk["stockout_rate"].fillna(0) >= 0.30)),
            (stock_risk["weekly_velocity"] > 0)
            & ((stock_risk["coverage_weeks"] < 8) | (stock_risk["stockout_rate"].fillna(0) >= 0.10)),
        ],
        ["high", "medium"],
        default="low",
    )
    save_table(stock_risk.sort_values("weekly_velocity", ascending=False), paths, "06_stock_risk_by_sku")

    plot_stock = stock_risk.query("weekly_velocity > 0").sort_values("weekly_velocity", ascending=False).head(800)
    if not plot_stock.empty:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.scatterplot(
            data=plot_stock,
            x="weekly_velocity",
            y="latest_stock_qty",
            hue="stock_risk",
            size="total_qty",
            sizes=(12, 120),
            alpha=0.65,
            ax=ax,
        )
        ax.set_xscale("symlog")
        ax.set_yscale("symlog")
        ax.set_title("Stock risk: velocity vs latest inventory")
        ax.set_xlabel("Weekly sales velocity")
        ax.set_ylabel(f"Latest stock qty ({latest_date.date()})")
        ax.legend(fontsize=8)
        save_figure(fig, paths, "06_stock_risk_velocity_matrix")

    sku = read_parquet(
        paths.input_dir,
        "sku",
        ["sku_cd", "line_desc", "categ_nm", "sub_categ_nm", "color_desc", "size", "launch_year", "launch_season"],
    )
    sales_segment = sales[["sku_cd", "region", "positive_qty", "net_amt"]].merge(sku, on="sku_cd", how="left")
    segment = (
        sales_segment.groupby(["region", "line_desc"], as_index=False)
        .agg(total_qty=("positive_qty", "sum"), net_amt=("net_amt", "sum"), sku_count=("sku_cd", "nunique"))
        .sort_values("total_qty", ascending=False)
    )
    save_table(segment, paths, "07_region_line_segment")
    top_lines = segment.groupby("line_desc")["total_qty"].sum().nlargest(12).index
    heat = segment[segment["line_desc"].isin(top_lines)].pivot_table(
        index="line_desc", columns="region", values="total_qty", aggfunc="sum", fill_value=0
    )
    if not heat.empty:
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(heat, cmap="Blues", ax=ax)
        ax.set_title("Cross-segment demand heatmap: region x line")
        ax.set_xlabel("Region")
        ax.set_ylabel("Line")
        save_figure(fig, paths, "07_region_line_segment_heatmap")

    return promo_effect, stock_risk, segment


def prescribe_actions(
    paths: Paths,
    demand_profile: pd.DataFrame,
    promo_effect: pd.DataFrame,
    stock_risk: pd.DataFrame,
) -> pd.DataFrame:
    strategy = demand_profile[["sku_cd", "demand_class", "abc_class", "total_qty", "eligible_min_24_weeks"]].merge(
        stock_risk[["sku_cd", "stock_risk", "coverage_weeks", "weekly_velocity"]], on="sku_cd", how="left"
    )
    strategy = strategy.merge(
        promo_effect[["sku_cd", "promo_lift"]], on="sku_cd", how="left"
    )
    strategy["promo_influence"] = pd.cut(
        strategy["promo_lift"],
        bins=[-np.inf, 0.10, 0.50, np.inf],
        labels=["low", "medium", "high"],
    ).astype("string").fillna("unknown")
    eligible = strategy["eligible_min_24_weeks"].fillna(False).astype(bool)
    weekly_velocity = strategy["weekly_velocity"].fillna(0)
    strategy["recommended_action"] = np.select(
        [
            (strategy["abc_class"] == "A") & (strategy["stock_risk"] == "high"),
            (strategy["promo_influence"] == "high") & (weekly_velocity > 0),
            strategy["demand_class"].isin(["Lumpy", "Intermittent"]) & eligible,
            ~eligible,
        ],
        [
            "replenish_or_rebalance_before_next_sales_cycle",
            "pre-position_inventory_before_promotion_window",
            "use_intermittent_or_hierarchical_forecast",
            "cold_start_attribute_pooling",
        ],
        default="monitor_with_weekly_baseline",
    )
    action_summary = (
        strategy.groupby(
            ["demand_class", "abc_class", "stock_risk", "promo_influence", "recommended_action"],
            as_index=False,
        )
        .agg(
            sku_count=("sku_cd", "nunique"),
            total_qty=("total_qty", "sum"),
            avg_weekly_velocity=("weekly_velocity", "mean"),
            avg_coverage_weeks=("coverage_weeks", "mean"),
        )
        .sort_values(["total_qty", "sku_count"], ascending=False)
    )
    save_table(strategy.sort_values("total_qty", ascending=False), paths, "08_prescriptive_actions_by_sku")
    save_table(action_summary, paths, "08_prescriptive_action_summary")

    top_action = action_summary.head(15)
    if not top_action.empty:
        fig, ax = plt.subplots(figsize=(11, 6))
        sns.barplot(data=top_action, y="recommended_action", x="total_qty", hue="stock_risk", dodge=False, ax=ax)
        ax.set_title("Prescriptive action priorities by demand quantity")
        ax.set_xlabel("Total demand quantity")
        ax.set_ylabel("")
        ax.legend(fontsize=8)
        save_figure(fig, paths, "08_prescriptive_action_priorities")
    return strategy


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    printable = df.reset_index()
    printable.columns = [str(c) for c in printable.columns]
    rows = []
    header = "| " + " | ".join(printable.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(printable.columns)) + " |"
    rows.extend([header, sep])
    for _, row in printable.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, float):
                values.append(f"{value:,.3f}")
            elif isinstance(value, (int, np.integer)):
                values.append(f"{value:,}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def metric_text(value: float) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.3f}"


def write_report(
    paths: Paths,
    overview: pd.DataFrame,
    sales: pd.DataFrame,
    demand_profile: pd.DataFrame,
    promo_effect: pd.DataFrame,
    stock_risk: pd.DataFrame,
    action_by_sku: pd.DataFrame,
    forecasts: pd.DataFrame,
    backtest_df: pd.DataFrame,
) -> Path:
    fig_dir = "../figures"
    date_min = sales["sales_date"].min().date().isoformat()
    date_max = sales["sales_date"].max().date().isoformat()
    valid_count = int(demand_profile["eligible_min_24_weeks"].sum())
    total_sku = demand_profile["sku_cd"].nunique()
    class_summary = (
        demand_profile.groupby("demand_class")
        .agg(sku_count=("sku_cd", "nunique"), total_qty=("total_qty", "sum"))
        .sort_values("total_qty", ascending=False)
    )
    class_md = markdown_table(class_summary)
    action_summary = (
        action_by_sku.groupby("recommended_action")
        .agg(sku_count=("sku_cd", "nunique"), total_qty=("total_qty", "sum"))
        .sort_values("total_qty", ascending=False)
        .head(10)
    )
    action_summary = markdown_table(action_summary)
    high_stock = stock_risk[stock_risk["stock_risk"] == "high"]["sku_cd"].nunique()
    promo_high = promo_effect[promo_effect["promo_lift"] >= 0.5]["sku_cd"].nunique()
    median_ma4 = backtest_df["ma4_wape"].median() if not backtest_df.empty else np.nan
    median_seasonal = backtest_df["seasonal_wape"].median() if not backtest_df.empty else np.nan

    report = f"""# MCM 최적화 및 의사결정 지원을 위한 Darts 기반 EDA 파이프라인

## 1. 분석 개요 및 비즈니스 목적

- 분석 목적: SKU 단위 판매 수요(`sales_qty`)의 간헐성, 변동성, 프로모션 민감도, 재고 커버리지를 통계적으로 분석하여 재고 전진 배치, 리밸런싱, 예측 모델 선택 기준을 수립한다.
- 분석 대상 및 기간: `{date_min}`부터 `{date_max}`까지의 MCM 판매 parquet 및 SKU, 재고, 프로모션 마스터.
- 목표 산출물: 데이터 정합성 검증, Darts `TimeSeries` 기반 주간 수요 구조 분석, ADI/CV2 수요 분류, 프로모션/재고/세그먼트 플롯, SKU별 처방 액션 테이블.

![dataset overview]({fig_dir}/01_dataset_overview_rows.png)

## 2. 데이터 정합성 검증 및 유효 구간 추출

- 원천 parquet: {len(overview):,}개 파일.
- 판매 행 수: {len(sales):,}건.
- 분석 SKU 수: {total_sku:,}개.
- 24주 이상 유효 수요 이력을 보유한 SKU: {valid_count:,}개.
- 유효 시작점은 SKU별 누적 판매량의 {VALID_CUM_QTY_PCT:.0%} 도달일로 정의해 초반 노이즈 구간을 제거했다.

![validation]({fig_dir}/03_validation_effective_history.png)

## 3. 통계적 데이터 프로파일링 및 패턴 분류

ADI 기준값 `{ADI_THRESHOLD}`, CV2 기준값 `{CV2_THRESHOLD}`를 적용해 Smooth, Erratic, Intermittent, Lumpy 수요군으로 분류했다.

{class_md}

![adi cv2]({fig_dir}/04_demand_profile_adi_cv2.png)

ABC/Pareto 관점에서는 상위 SKU가 전체 수요의 대부분을 견인하므로, 모델링과 재고 의사결정은 A등급 SKU를 우선 대상으로 삼는 것이 효율적이다.

![abc pareto]({fig_dir}/04_abc_pareto_curve.png)

## 4. Darts TimeSeries 구조 분석 및 예측 기준선

전체 일별 수요를 주간 수요로 집계한 뒤 Darts `TimeSeries`로 변환했다. 이 객체를 기준으로 주간 수요 시각화, 상위 SKU baseline forecast, holdout backtest를 수행했다.

![weekly]({fig_dir}/05_darts_weekly_total_series.png)

STL 분해 결과는 추세, 계절성, 잔차 이상 구간을 분리해 이벤트성 급등락을 추적할 수 있게 한다.

![stl]({fig_dir}/05_stl_decomposition.png)

상위 SKU에 대해서는 Darts 시계열을 기반으로 최근 이동평균과 52주 계절 naive를 결합한 12주 baseline forecast를 산출했다.

![forecast]({fig_dir}/05_top_sku_darts_forecasts.png)

최근 52주 수요가 있는 상위 SKU holdout backtest의 중앙 WAPE는 MA4 `{metric_text(median_ma4)}`, seasonal naive `{metric_text(median_seasonal)}`이다.

![backtest]({fig_dir}/05_darts_baseline_backtest.png)

## 5. 다변량/구조적 동인 분석

프로모션 기간 내 평균 일 판매량과 비프로모션 평균 일 판매량을 비교해 SKU별 lift를 산출했다. Lift가 50% 이상인 상위 SKU는 {promo_high:,}개로, 프로모션 사전 재고 배치의 우선 후보가 된다.

![promo]({fig_dir}/06_promotion_lift_top_skus.png)

재고 리스크는 최신 재고, 최근 12주 판매 속도, stockout snapshot 비율을 결합해 high/medium/low로 구분했다. High risk SKU는 {high_stock:,}개다.

![stock]({fig_dir}/06_stock_risk_velocity_matrix.png)

## 6. 다차원 세그먼트 교차 분석

지역과 상품 line의 교차 수요를 heatmap으로 비교했다. 특정 line이 특정 region에 집중되는 구조는 리밸런싱, 지역별 캠페인, 사이즈/컬러 배분 정책의 근거가 된다.

![segment]({fig_dir}/07_region_line_segment_heatmap.png)

## 7. 최종 비즈니스 처방 및 액션 플랜

SKU별로 수요군, ABC, 재고 리스크, 프로모션 lift를 결합해 실행 액션을 배정했다.

{action_summary}

![action]({fig_dir}/08_prescriptive_action_priorities.png)

### 실행 권고

- A등급이면서 stock risk가 high인 SKU는 다음 판매 사이클 이전에 재고 보충 또는 지역 간 재배치를 우선 실행한다.
- 프로모션 lift가 high인 SKU는 프로모션 시작 전 최소 1~2개 주차 앞서 목표 채널에 재고를 전진 배치한다.
- Intermittent/Lumpy SKU는 일반 회귀 모델보다 intermittent method, weekly aggregation, hierarchical pooling을 우선 검토한다.
- 24주 미만 이력 SKU는 단독 시계열 모델보다 line/category/color/size 속성 기반 cold-start pooling으로 예측한다.

## 산출물

- `tables/03_valid_history_by_sku.csv`
- `tables/04_demand_profile_by_sku.csv`
- `tables/05_top_sku_darts_forecasts.csv`
- `tables/06_promotion_lift_top_skus.csv`
- `tables/06_stock_risk_by_sku.csv`
- `tables/08_prescriptive_actions_by_sku.csv`
"""
    report_path = paths.reports_dir / "MCM_DARTS_EDA_REPORT.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    paths = init_paths(args.input.resolve(), args.output.resolve())
    sns.set_theme(style="whitegrid", font="DejaVu Sans")

    overview = build_dataset_overview(paths)
    sales, daily_sku = build_sales_base(paths)
    valid_stats = validate_effective_history(paths, daily_sku)
    demand_profile = profile_demand(paths, daily_sku, valid_stats)
    forecasts, backtest_df = structural_darts_analysis(
        paths, daily_sku, demand_profile, args.forecast_horizon, args.top_sku_count
    )
    promo_effect, stock_risk, _segment = promotion_stock_segment_analysis(
        paths, sales, daily_sku, demand_profile
    )
    action_by_sku = prescribe_actions(paths, demand_profile, promo_effect, stock_risk)
    report_path = write_report(
        paths,
        overview,
        sales,
        demand_profile,
        promo_effect,
        stock_risk,
        action_by_sku,
        forecasts,
        backtest_df,
    )
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
