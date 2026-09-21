# Freight Rate Prediction (Spotter ML assessment)

Predicts `posted_rate` for 12,000 validation loads with a LightGBM gradient-boosting model, validated on a
chronological holdout and compared against a lane-median baseline.

## Setup

```bash
python -m pip install -r requirements.txt
```

Put the assessment files in a `data/` folder next to `Spotter.py` (they are not committed):

```
data/train_test.csv
data/validation.csv
data/validation_predictions_template.csv
data/december_chart_inputs.csv
```

Dash spellings (`train-test.csv`) also work. On Kaggle, attach the files as a dataset; the script finds them under
`/kaggle/input` and writes outputs to `/kaggle/working`. `SPOTTER_DATA_DIR` / `SPOTTER_OUTPUT_DIR` override both.

## Run

```bash
python Spotter.py        # trains, validates, writes the prediction files
python score.py --predictions validation_predictions.csv --december-predictions december_chart_inputs.csv
python make_report.py    # builds Freight_Rate_Report.pdf from results.json and the two charts
```

`score.py` is the scorer supplied with the assessment (not included here).

## Outputs

| File | Purpose |
|---|---|
| `validation_predictions.csv` | `load_id,predicted_rate`, same IDs and order as the template |
| `december_chart_inputs.csv` | December input file with `predicted_rate` filled in (README step 4) |
| `model.joblib` | The trained model and its preprocessing; every prediction comes from it |
| `results.json` | Metrics and settings used by the report |
| `validation_diagnostics.png` | Predicted vs actual, error distribution, loss curves |
| `holdout_worst_errors.csv` | The 20 holdout rows with the largest errors |
| `data_quality_notes.md`, `report.md` | Data checks and a short text summary |
| `Freight_Rate_Report.pdf` | Report with split approach and the December chart |

## Method

1. **Clean:** missing coordinates are restored from other rows naming the same place; remaining gaps are imputed
   (fitted on the training partition only). Duplicates are reported, not deleted.
2. **Split:** training rows are ordered by date; the first 80% fit the model and the last 20% form the holdout,
   because the validation set is a later period.
3. **Features:** supplied columns (one-hot for categories) plus calendar features (day of week, day of month, month,
   days to and since the nearest US federal holiday). December inputs have no market columns, so those are filled
   with the typical value from the last 30 days of labelled data.
4. **Model:** LightGBM with early stopping on the holdout; the final model is refit on all labelled rows with the
   round count found. Predictions are floored at the smallest positive training rate (the scorer rejects rates <= 0).
5. **Evaluation:** RMSE and MAE on the holdout, against a lane-median baseline, on all rows and on typical rows.

## Configuration (top of `Spotter.py`)

- `DROP_RATE_OUTLIERS` (default `True`, the setting used for the submitted predictions): drop training rows whose rate per mile is far outside the normal range (Tukey fence,
  `RPM_FENCE`). Holdout and validation rows are never dropped. The rule and row count are written to the notes.
- `LEARNING_RATE`, `MAX_ROUNDS`, `EARLY_STOPPING_ROUNDS`, `HOLDOUT_FRACTION`, `SEED`.

## Limitations

- Training data ends before November, while validation includes November and December, so seasonality there cannot
  be learned directly.
- A small share of holdout rows have extreme rates and dominate overall RMSE; both all-row and typical-row numbers
  are reported.
