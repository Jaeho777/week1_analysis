# MCM Darts Analysis

Fresh EDA and decision-support pipeline for the MCM parquet datasets, built
inside the Darts repository.

## Run

From `C:\Users\user\Downloads\0706데이터분석`:

```powershell
& 'C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\darts\mcm_darts_analysis\run_mcm_darts_pipeline.py --input .\mcm --output .\darts\mcm_darts_analysis\output
```

The script reads the parquet originals directly and rewrites the output folder
on every run.

## What It Produces

- `output/reports/MCM_DARTS_EDA_REPORT.md`
- `output/figures/*.png`
- `output/tables/*.csv`

## Analysis Blocks

- Data validation and effective demand-history filtering
- ADI/CV2 demand classification and ABC/Pareto profiling
- Darts `TimeSeries` weekly demand construction
- STL structural decomposition
- Top-SKU baseline forecast and holdout backtest
- Promotion lift analysis
- Stock-risk and sales-velocity matrix
- Region x product-line cross-segment heatmap
- SKU-level prescriptive action assignment

## Current Modeling Scope

This first pass intentionally uses Darts core `TimeSeries` objects plus
statistical baselines. Deep learning and heavier optional model families can be
added after the target SKU/channel granularity is confirmed.
