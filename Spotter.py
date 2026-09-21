"""Freight-rate prediction pipeline (Spotter ML assessment).

Model: LightGBM gradient boosting. It has no epochs; it adds decision trees
("boosting rounds") one at a time, and early stopping on a chronological
holdout picks how many.

Run locally / Colab:  python Spotter.py
Run on Kaggle:        add the CSV files as a (private) Dataset, attach it to
                      the notebook, then run this file as one cell or with
                      !python Spotter.py   (files are found under /kaggle/input,
                      outputs are written to /kaggle/working)

Input files (dash or underscore spelling both work):
    train_test.csv, validation.csv, december_chart_inputs.csv

Outputs:
    validation_predictions.csv   load_id,predicted_rate (same IDs and order as the template)
    december_chart_inputs.csv    the December input file with predicted_rate filled in (README step 4)
    model.joblib                 the trained model + preprocessing (all predictions come from it)
    results.json                 all numbers used by make_report.py
    validation_diagnostics.png   predicted vs actual, error histogram, loss curves
    data_quality_notes.md        data checks and the fixes applied
    report.md                    split approach and holdout results

Then score with:
    python score.py --predictions validation_predictions.csv \
                    --december-predictions december_chart_inputs.csv
    python make_report.py          # builds the PDF report from results.json + the two charts
"""

import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")
# Optional overrides: SPOTTER_DATA_DIR / SPOTTER_OUTPUT_DIR environment variables.
OUT_DIR = Path(os.environ.get("SPOTTER_OUTPUT_DIR", KAGGLE_WORKING if KAGGLE_WORKING.exists() else ROOT))
DATA_DIRS = [Path(p) for p in [os.environ.get("SPOTTER_DATA_DIR")] if p] + [ROOT / "data", ROOT, OUT_DIR]

TARGET = "posted_rate"
ID = "load_id"
DATE = "date"
NON_FEATURES = [ID, TARGET, DATE, "predicted_rate"]

HOLDOUT_FRACTION = 0.20   # most recent 20% of training rows, by date
LEARNING_RATE = 0.05
MAX_ROUNDS = 3000         # upper limit; early stopping picks the real number
EARLY_STOPPING_ROUNDS = 100
SEED = 42

# About 1.4% of rows have rates far outside the normal range, spread evenly across every
# feature, so nothing explains them. True (the final setting) drops those rows from TRAINING
# only. Holdout and validation rows are never dropped. Set False to train on the raw data.
DROP_RATE_OUTLIERS = True
RPM_FENCE = 3.0           # Tukey fence: Q1 - 3*IQR .. Q3 + 3*IQR of rate per mile

COORDINATES = {
    "pickup": ["pickup_lat", "pickup_lon"],
    "delivery": ["delivery_lat", "delivery_lon"],
}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def find_file(stem: str, required: bool = True):
    """Find a data file: beside the script, in ./data, or anywhere under /kaggle/input."""
    names = {f"{stem}.csv", f"{stem.replace('_', '-')}.csv"}
    for folder in DATA_DIRS:
        for name in names:
            if (folder / name).exists():
                return folder / name
    if KAGGLE_INPUT.exists():
        for name in names:
            match = next(KAGGLE_INPUT.rglob(name), None)
            if match is not None:
                return match
    if not required:
        return None
    raise FileNotFoundError(f"Could not find any of {sorted(names)} (looked in {ROOT}, "
                            f"{ROOT / 'data'} and {KAGGLE_INPUT})")


def parse_dates(series: pd.Series) -> pd.Series:
    """pandas.to_datetime; if some dates fail because formats are mixed, retry format-tolerant."""
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().sum() > series.isna().sum():
        try:
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
        except (TypeError, ValueError):
            pass
    return parsed


def load_data():
    train = pd.read_csv(find_file("train_test"))
    validation = pd.read_csv(find_file("validation"))
    december_raw = pd.read_csv(find_file("december_chart_inputs"))  # written back untouched + predictions
    december = december_raw.copy()
    for frame in (train, validation, december):
        frame[DATE] = parse_dates(frame[DATE])
    return train, validation, december, december_raw


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def restore_coordinates(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Fill missing coordinates from other rows that name the same place.

    Uses the first NON-missing coordinate seen for each place in the labelled
    data. (Dropping duplicates instead would keep the first row per place, which
    can itself be missing.) Coordinates are facts about a place, not about the
    target, so using the full training file as the lookup does not leak labels.
    """
    data = frame.copy()
    for place, columns in COORDINATES.items():
        lookup = reference.groupby(place)[columns].first()
        for column in columns:
            filled = data[place].map(lookup[column])
            data[column] = data[column].fillna(filled) if column in data else filled
    return data


HOLIDAYS = USFederalHolidayCalendar().holidays(start="2020-01-01", end="2030-12-31").values.astype("datetime64[D]")


def add_date_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calendar features. The December chart varies ONLY by date, so the model needs these.

    Training data ends before November, so the model has never seen Thanksgiving or
    Christmas. Distance to the nearest US federal holiday lets it carry over whatever it
    learns from holidays it HAS seen (Memorial Day, July 4, Labor Day...).
    """
    data = frame.copy()
    dates = data[DATE].fillna(data[DATE].median())
    days = dates.values.astype("datetime64[D]")
    next_holiday = HOLIDAYS[np.minimum(np.searchsorted(HOLIDAYS, days, side="left"), len(HOLIDAYS) - 1)]
    last_holiday = HOLIDAYS[np.maximum(np.searchsorted(HOLIDAYS, days, side="right") - 1, 0)]
    data["dow"] = dates.dt.dayofweek
    data["dom"] = dates.dt.day
    data["month"] = dates.dt.month
    data["days_to_holiday"] = np.clip((next_holiday - days).astype(int), 0, 30)
    data["days_since_holiday"] = np.clip((days - last_holiday).astype(int), 0, 30)
    return data


CALENDAR_FEATURES = ("dow", "dom", "month", "days_to_holiday", "days_since_holiday")


def prepare_features(frame: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Model inputs: calendar features + everything except ID, date, target, prediction."""
    features = add_date_features(frame).drop(columns=NON_FEATURES, errors="ignore")
    return features if columns is None else features.reindex(columns=columns)


def fill_missing_context(frame: pd.DataFrame, train: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Give the December rows any model feature they do not carry.

    The December input has no market_index / quote_signal style columns. Rather than
    letting them fall back to the all-time median, use the typical value from the most
    recent 30 days of labelled data.
    """
    data = frame.copy()
    recent = train[train[DATE] >= train[DATE].max() - pd.Timedelta(days=30)]
    for column in columns:
        if column not in data.columns:
            series = recent[column]
            data[column] = series.median() if pd.api.types.is_numeric_dtype(series) else series.mode().iloc[0]
    return data


def quality_report(train, validation, december, missing_before, missing_after) -> list[str]:
    """Computed data checks. Everything printed here is measured, not assumed."""
    lines = []
    for name, frame in [("train", train), ("validation", validation), ("december", december)]:
        missing = frame.drop(columns="predicted_rate", errors="ignore").isna().sum()
        lines += [
            f"{name}: {len(frame):,} rows, {frame.shape[1]} columns",
            f"  date range: {frame[DATE].min().date()} to {frame[DATE].max().date()} "
            f"(unparseable dates: {int(frame[DATE].isna().sum())})",
            f"  duplicate rows: {int(frame.duplicated().sum())}; "
            f"duplicate load IDs: {int(frame[ID].duplicated().sum()) if ID in frame else 'n/a (no load_id column)'}",
            "  missing: " + (", ".join(f"{k}={v}" for k, v in missing[missing > 0].items()) or "none"),
            f"  non-positive distance: {int((frame['distance'] <= 0).sum())}; "
            f"non-positive weight: {int((frame['weight'] <= 0).sum())}",
        ]
    rate = train[TARGET]
    lines += [
        f"target {TARGET}: min={rate.min():.2f}, median={rate.median():.2f}, "
        f"max={rate.max():.2f}; non-positive={int((rate <= 0).sum())}",
        "target quantiles 1% / 50% / 99% / 99.9%: "
        + " / ".join(f"{v:,.1f}" for v in rate.quantile([0.01, 0.5, 0.99, 0.999])),
        "rate per mile quantiles 0.1% / 1% / 50% / 99% / 99.9%: "
        + " / ".join(f"{v:.2f}" for v in (rate / train["distance"]).replace([np.inf, -np.inf], np.nan)
                     .quantile([0.001, 0.01, 0.5, 0.99, 0.999])),
        "median rate per mile by weekday (Mon..Sun): "
        + ", ".join(f"{v:.2f}" for v in rate_per_mile(train).groupby(train[DATE].dt.dayofweek).median()),
        "median rate per mile by month: "
        + ", ".join(f"{m}:{v:.2f}" for m, v in rate_per_mile(train).groupby(train[DATE].dt.month).median().items()),
        *outlier_profile(train),
        f"equipment only in validation: "
        f"{sorted(set(validation['equipment'].dropna()) - set(train['equipment'].dropna())) or 'none'}",
    ]
    for place, columns in COORDINATES.items():
        inconsistent = int((train.groupby(place)[columns[0]].nunique() > 1).sum())
        lines.append(f"places with more than one {place} latitude in train: {inconsistent}")
    lines += [
        "missing coordinates before restoring -> after: "
        + ", ".join(f"{name}: {missing_before[name]} -> {missing_after[name]}" for name in missing_before),
        "",
        "Fixes applied:",
        "missing coordinates are restored from the same place elsewhere in the training data; "
        "remaining numeric gaps are median-imputed and categorical gaps mode-imputed "
        "(both fitted on the training partition only); unseen categories are ignored by the encoder.",
        "Duplicates are reported, not deleted, because repeated loads can be valid observations.",
    ]
    return lines


def format_note(line: str) -> str:
    """Markdown bullets: indented lines become sub-bullets, blank lines stay blank."""
    if not line:
        return ""
    return f"  - {line.strip()}" if line.startswith("  ") else f"- {line}"


# ----------------------------------------------------------------------------
# Modelling
# ----------------------------------------------------------------------------
def build_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        remainder="drop",
    )


def build_model(n_estimators: int) -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=LEARNING_RATE,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.85,
        subsample_freq=1,          # without this, subsample is silently ignored
        colsample_bytree=0.85,
        reg_lambda=2.0,
        objective="regression",
        metric="rmse",
        random_state=SEED,
        n_jobs=-1,
        verbosity=-1,
    )


def evaluate(actual, predicted) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
    }


def error_diagnostics(holdout: pd.DataFrame, predicted: np.ndarray, path: Path) -> list[str]:
    """Where does the holdout error come from? Saves the 20 worst rows to a CSV."""
    frame = holdout.assign(predicted=predicted)
    frame["error"] = frame[TARGET] - frame["predicted"]
    frame["abs_error"] = frame["error"].abs()
    worst = frame.nlargest(20, "abs_error")
    worst.to_csv(path, index=False)
    shown = [c for c in ["date", "pickup", "delivery", "distance", "equipment", TARGET, "predicted"] if c in worst]
    print("Worst 10 holdout errors:\n" + worst.head(10)[shown].to_string(
        index=False, float_format=lambda v: f"{v:,.1f}"))
    squared = frame["error"] ** 2
    top_share = squared.nlargest(max(1, len(frame) // 100)).sum() / squared.sum()
    return [
        f"Holdout: the worst 1% of rows account for {top_share:.0%} of the squared error "
        f"(RMSE {np.sqrt(squared.mean()):.1f} vs MAE {frame['abs_error'].mean():.1f}).",
        f"Median absolute error: {frame['abs_error'].median():.1f}; "
        f"worst single error: {frame['abs_error'].max():,.1f} (see holdout_worst_errors.csv).",
    ]


def importance_summary(model: LGBMRegressor, preprocessor: ColumnTransformer) -> list[str]:
    """Which inputs does the model actually use? Answers 'why is December flat?'."""
    names = [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
    gain = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=names)
    gain = gain / gain.sum()
    calendar = float(gain[gain.index.isin(CALENDAR_FEATURES)].sum())
    top = ", ".join(f"{name} {share:.0%}" for name, share in gain.nlargest(8).items())
    return [f"Calendar features' share of model gain: {calendar:.1%}.", f"Top features by gain: {top}."]


def outlier_profile(train: pd.DataFrame) -> list[str]:
    """Are the extreme-rate rows random junk, or do they follow a pattern the model could learn?"""
    fences = rate_fences(train)
    extreme = ~is_typical(train, fences)
    lines = [f"extreme rows (rate per mile outside {fences[0]:.2f}..{fences[1]:.2f}): "
             f"{int(extreme.sum()):,} of {len(train):,} ({extreme.mean():.1%})"]
    by_equipment = extreme.groupby(train["equipment"]).mean()
    lines.append("extreme rows, share of each equipment type: "
                 + ", ".join(f"{k} {v:.1%}" for k, v in by_equipment.items()))
    by_month = extreme.groupby(train[DATE].dt.month).mean()
    lines.append("extreme rows, share by month: " + ", ".join(f"{m}:{v:.1%}" for m, v in by_month.items()))
    for column in [c for c in ("distance", "weight", "market_index", "quote_signal") if c in train]:
        lines.append(f"extreme rows, median {column}: typical {train.loc[~extreme, column].median():,.2f} "
                     f"vs extreme {train.loc[extreme, column].median():,.2f}")
    multiple = (rate_per_mile(train)[extreme] / rate_per_mile(train)[~extreme].median())
    lines.append(f"extreme rows, rate per mile as a multiple of the typical median: "
                 f"median {multiple.median():.1f}x (10th-90th pct {multiple.quantile(.1):.1f}x-{multiple.quantile(.9):.1f}x)")
    return lines


def rate_per_mile(frame: pd.DataFrame) -> pd.Series:
    return (frame[TARGET] / frame["distance"]).replace([np.inf, -np.inf], np.nan)


def rate_fences(frame: pd.DataFrame) -> tuple[float, float]:
    rpm = rate_per_mile(frame)
    q1, q3 = rpm.quantile([0.25, 0.75])
    return q1 - RPM_FENCE * (q3 - q1), q3 + RPM_FENCE * (q3 - q1)


def is_typical(frame: pd.DataFrame, fences: tuple[float, float]) -> pd.Series:
    return rate_per_mile(frame).between(*fences)


def lane_median_baseline(fit_part: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """Yardstick: median rate per pickup->delivery lane (overall median if unseen)."""
    def lane(frame):
        return frame["pickup"].fillna("Unknown") + " -> " + frame["delivery"].fillna("Unknown")

    medians = fit_part[TARGET].groupby(lane(fit_part)).median()
    return lane(holdout).map(medians).fillna(fit_part[TARGET].median()).to_numpy()


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_diagnostics(actual, predicted, evals, best_round, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(actual, predicted, s=8, alpha=0.35)
    low, high = float(np.min(actual)), float(np.max(actual))
    axes[0].plot([low, high], [low, high], color="black", linewidth=1)
    axes[0].set(xlabel="Actual rate", ylabel="Predicted rate", title="Predicted vs actual (holdout)")

    axes[1].hist(actual - predicted, bins=40, edgecolor="white")
    axes[1].set(xlabel="Actual - predicted", ylabel="Count", title="Error distribution (holdout)")

    rounds = np.arange(1, len(evals["train"]["rmse"]) + 1)
    axes[2].plot(rounds, evals["train"]["rmse"], label="train")
    axes[2].plot(rounds, evals["holdout"]["rmse"], label="holdout")
    axes[2].axvline(best_round, color="gray", linestyle="--", label=f"best round ({best_round})")
    axes[2].set(xlabel="Boosting round", ylabel="RMSE", title="Training and holdout loss")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    # 1. Load
    train, validation, december, december_raw = load_data()

    # 2. Clean: restore missing coordinates in every file, then record the checks
    coordinate_columns = [c for cols in COORDINATES.values() for c in cols]
    missing_before = {name: int(frame.reindex(columns=coordinate_columns).isna().sum().sum())
                      for name, frame in [("train", train), ("validation", validation), ("december", december)]}
    reference = train.copy()
    train = restore_coordinates(train, reference)
    validation = restore_coordinates(validation, reference)
    december = restore_coordinates(december, reference)
    missing_after = {name: int(frame.reindex(columns=coordinate_columns).isna().sum().sum())
                     for name, frame in [("train", train), ("validation", validation), ("december", december)]}

    notes = quality_report(train, validation, december, missing_before, missing_after)

    # 3. Split chronologically: validation is a later period than training
    train = train.sort_values(DATE).reset_index(drop=True)
    split_at = int(len(train) * (1 - HOLDOUT_FRACTION))
    fit_part, holdout = train.iloc[:split_at], train.iloc[split_at:]
    fences = rate_fences(fit_part)
    typical_holdout = is_typical(holdout, fences).to_numpy()
    dropped = 0
    if DROP_RATE_OUTLIERS:
        keep = is_typical(fit_part, fences)
        dropped = int((~keep).sum())
        fit_part = fit_part[keep]
    x_fit, y_fit = prepare_features(fit_part), fit_part[TARGET]
    x_holdout, y_holdout = prepare_features(holdout), holdout[TARGET]

    # 4. Encode: fit the preprocessing on the fit partition only
    numeric = x_fit.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in x_fit.columns if c not in numeric]
    preprocessor = build_preprocessor(categorical, numeric)
    x_fit_encoded = preprocessor.fit_transform(x_fit)
    x_holdout_encoded = preprocessor.transform(x_holdout)

    # 5. Train with early stopping on the chronological holdout
    # When outlier rows are being dropped, early stopping must not be steered by them either:
    # watch the typical holdout rows. (Predictions and the final report still cover every row.)
    watch = typical_holdout if DROP_RATE_OUTLIERS else np.ones(len(holdout), dtype=bool)
    model = build_model(MAX_ROUNDS)
    model.fit(
        x_fit_encoded, y_fit,
        eval_set=[(x_fit_encoded, y_fit), (x_holdout_encoded[watch], y_holdout[watch])],
        eval_names=["train", "holdout"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    best_round = int(model.best_iteration_)
    holdout_pred = model.predict(x_holdout_encoded)

    # 6. Test against the lane-median baseline
    results = evaluate(y_holdout, holdout_pred)
    baseline_pred = lane_median_baseline(fit_part, holdout)
    baseline_results = evaluate(y_holdout, baseline_pred)
    typical_results = evaluate(y_holdout[typical_holdout], holdout_pred[typical_holdout])
    typical_baseline = evaluate(y_holdout[typical_holdout], baseline_pred[typical_holdout])
    gain = 100 * (1 - results["RMSE"] / baseline_results["RMSE"])
    mae_gain = 100 * (1 - results["MAE"] / baseline_results["MAE"])
    error_notes = error_diagnostics(holdout, holdout_pred, OUT_DIR / "holdout_worst_errors.csv")
    error_notes += importance_summary(model, preprocessor)
    error_notes.append(
        f"Typical holdout rows ({int(typical_holdout.sum()):,} of {len(holdout):,}, rate per mile inside "
        f"{fences[0]:.2f}..{fences[1]:.2f}): LightGBM RMSE {typical_results['RMSE']:.1f} / MAE "
        f"{typical_results['MAE']:.1f} vs baseline RMSE {typical_baseline['RMSE']:.1f} / MAE "
        f"{typical_baseline['MAE']:.1f}.")
    if DROP_RATE_OUTLIERS:
        error_notes.append(
            f"Outlier rule: {dropped} training rows dropped because rate per mile fell outside "
            f"{fences[0]:.2f}..{fences[1]:.2f} (Q1 - {RPM_FENCE:g}*IQR .. Q3 + {RPM_FENCE:g}*IQR of the fit "
            "partition); treated as noisy labels that no feature explains. Early stopping watched typical holdout rows only. "
            "Holdout and validation rows are never dropped.")
    else:
        error_notes.append("No training rows dropped (DROP_RATE_OUTLIERS=False).")

    # 7. Plot
    plot_diagnostics(y_holdout, holdout_pred, model.evals_result_, best_round,
                     OUT_DIR / "validation_diagnostics.png")

    # 8. Final fit on all labelled rows, using the round count found above
    final_train = train[is_typical(train, rate_fences(train))] if DROP_RATE_OUTLIERS else train
    full_x = prepare_features(final_train)
    model_columns = full_x.columns.tolist()
    final_preprocessor = build_preprocessor(categorical, numeric)
    final_model = build_model(best_round)
    final_model.fit(final_preprocessor.fit_transform(full_x), final_train[TARGET])
    # Keep the trained model: every validation and December prediction below comes from it.
    joblib.dump({"preprocessor": final_preprocessor, "model": final_model, "columns": model_columns},
                OUT_DIR / "model.joblib")

    rate_floor = float(train.loc[train[TARGET] > 0, TARGET].min())  # score.py rejects rates <= 0

    def predict(frame: pd.DataFrame) -> np.ndarray:
        raw = final_model.predict(final_preprocessor.transform(prepare_features(frame, model_columns)))
        return np.clip(raw, rate_floor, None)

    # 9. Write outputs. December keeps its original seven columns (score.py insists on
    #    that); the original input file is never modified.
    submission = pd.DataFrame({ID: validation[ID], "predicted_rate": predict(validation)})
    template_path = find_file("validation_predictions_template", required=False)
    if template_path is not None:  # follow the supplied template exactly: same IDs, same order
        order = pd.read_csv(template_path)[ID]
        if set(order) != set(submission[ID]) or len(order) != len(submission):
            raise ValueError("validation.csv and the predictions template do not contain the same load_id values")
        submission = submission.set_index(ID).loc[order].reset_index()
    submission.to_csv(OUT_DIR / "validation_predictions.csv", index=False)

    december_missing = [c for c in model_columns if c not in add_date_features(december).columns]
    december_pred = predict(fill_missing_context(december, train, december_missing))
    december_out = december_raw.copy()
    december_out["predicted_rate"] = december_pred
    december_out.to_csv(OUT_DIR / "december_chart_inputs.csv", index=False)
    december_spread = float(december_pred.max() - december_pred.min())

    notes += error_notes + [
        f"December inputs lack these model features, so they were filled with the typical value "
        f"from the last 30 days of labelled data: {december_missing or 'none'}.",
        f"Split: validation is a later period than training, so training rows were ordered by date; "
        f"the first {1 - HOLDOUT_FRACTION:.0%} fit the model and the last {HOLDOUT_FRACTION:.0%} form the holdout.",
    ]
    (OUT_DIR / "data_quality_notes.md").write_text(
        "# Data-quality notes\n\n" + "\n".join(format_note(line) for line in notes) + "\n",
        encoding="utf-8")

    report = [
        "# Freight rate prediction report",
        "",
        "## Split and validation",
        f"Training rows were sorted by date. The first {1 - HOLDOUT_FRACTION:.0%} fit the model and the most "
        f"recent {HOLDOUT_FRACTION:.0%} were held out, because the supplied validation set is a later period. "
        "Encoders and imputers were fitted on the fit partition only. "
        f"Boosting stopped at round {best_round} using the holdout loss; the final model was refit on all "
        "labelled rows with that number of rounds.",
        "",
        "## Features",
        "Existing columns plus calendar features (day of week, day of month, month, days to and since the "
        "nearest US federal holiday). The December chart "
        "varies only by date, so calendar features are what let the model respond to it. December inputs "
        f"lack {december_missing or 'no'} model feature(s); those were filled with recent typical values.",
        "",
        "## Holdout results",
        f"- LightGBM: RMSE {results['RMSE']:.3f}, MAE {results['MAE']:.3f}",
        f"- Lane-median baseline: RMSE {baseline_results['RMSE']:.3f}, MAE {baseline_results['MAE']:.3f}",
        f"- Improvement over baseline: RMSE {gain:.1f}%, MAE {mae_gain:.1f}%",
        *[f"- {line}" for line in error_notes],
        "",
        "## Files",
        "- `validation_diagnostics.png`: predicted vs actual, error distribution, loss curves",
        "- `validation_predictions.csv`: final validation predictions",
        "- `december_chart_inputs.csv`: December inputs with predicted_rate filled in, for the scorer's chart",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    results_json = {
        "holdout": results, "baseline": baseline_results,
        "typical_holdout": typical_results, "typical_baseline": typical_baseline,
        "rmse_gain_pct": gain, "mae_gain_pct": mae_gain,
        "best_round": best_round, "rounds_trained": int(len(model.evals_result_["train"]["rmse"])),
        "learning_rate": LEARNING_RATE, "drop_rate_outliers": DROP_RATE_OUTLIERS,
        "rate_per_mile_fences": [float(fences[0]), float(fences[1])], "rows_dropped": dropped,
        "rows": {"train_total": int(len(train)), "fit": int(len(fit_part)), "holdout": int(len(holdout)),
                 "typical_holdout": int(typical_holdout.sum()), "validation": int(len(validation))},
        "dates": {"train": [str(train[DATE].min().date()), str(train[DATE].max().date())],
                  "holdout": [str(holdout[DATE].min().date()), str(holdout[DATE].max().date())],
                  "validation": [str(validation[DATE].min().date()), str(validation[DATE].max().date())]},
        "december_range": [float(december_pred.min()), float(december_pred.max())],
        "notes": notes,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results_json, indent=2), encoding="utf-8")

    train_loss = np.asarray(model.evals_result_["train"]["rmse"])
    holdout_loss = np.asarray(model.evals_result_["holdout"]["rmse"])
    scope = " (typical rows only)" if DROP_RATE_OUTLIERS else ""
    print(f"Final train RMSE: {train_loss[-1]:.3f}")
    print(f"Final holdout RMSE{scope}: {holdout_loss[-1]:.3f}")
    print(f"Best holdout RMSE{scope}: {holdout_loss[best_round - 1]:.3f} at round {best_round}")
    print(f"Rounds trained: {len(train_loss)} (early stopping; predictions use round {best_round})")
    print("Holdout:", results)
    print("Lane median:", baseline_results)
    print(f"Improvement over baseline: RMSE {gain:.1f}%, MAE {mae_gain:.1f}%")
    for line in error_notes:
        print(line)
    for line in notes:
        if line.startswith(("median rate per mile by", "extreme rows")):
            print(line)
    print(f"Train dates {train[DATE].min().date()} -> {train[DATE].max().date()}; "
          f"validation {validation[DATE].min().date()} -> {validation[DATE].max().date()}")
    print(f"December prediction range: {december_pred.min():.2f} to {december_pred.max():.2f}")
    if december_spread < 1e-6:
        print("WARNING: December predictions are flat - the model is ignoring the date.")
    print("Created validation_predictions.csv, december_chart_inputs.csv, model.joblib, results.json, validation_diagnostics.png, "
          "holdout_worst_errors.csv, data_quality_notes.md, report.md")


if __name__ == "__main__":
    main()
