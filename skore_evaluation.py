"""skore evaluation of the fuel-poverty risk classifiers (Pipelines A and B).

Evaluates the models trained in `Fuel_Poverty_Python.ipynb` (Steps 10-14) with
skore's EstimatorReport / ComparisonReport instead of hand-rolled metrics, and
writes the full output to SKORE_EVALUATION.md.

The data preparation below is a faithful reproduction of the notebook's
Step 10 (rename / transform / cap / dummy-encode / SMOTE) and Step 08
(stratified 80/20 split, seed 42) so the models are evaluated on exactly the
data the pipeline produces. Model hyperparameters mirror Steps 11/12/14.
The DNN (Step 13) is excluded, matching the notebook's own Step 15.

Usage:
    python skore_evaluation.py [--data /path/to/ml_model_data_with_predictions.parquet]

skore API verified against skore 0.22.0 (EstimatorReport / ComparisonReport /
metrics.summarize().frame() / to_markdown()).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from skore import ComparisonReport, EstimatorReport
from xgboost import XGBClassifier

DEFAULT_DATA = Path("/Users/hayder/Desktop/R Projects/ML_Project/ml_model_data_with_predictions.parquet")
OUTPUT_MD = Path(__file__).resolve().parent / "SKORE_EVALUATION.md"


# --------------------------------------------------------------------------
# Data preparation — mirrors notebook Step 10 (preprocessing + SMOTE) and
# Step 08 (stratified split). Logic unchanged from the pipeline.
# --------------------------------------------------------------------------
def prepare_data(data_path: Path) -> dict:
    df = pd.read_parquet(data_path)
    n_raw = len(df)

    # Step 10: renames (non-strict, as in the notebook conversion)
    df = df.rename(columns={
        "CURRENT_ENERGY_EFFICIENCY": "current_energy_efficiency",
        "ENERGY_CONSUMPTION_CURRENT": "energy_consumption_current",
        "CO2_EMISSIONS_CURRENT": "co2_emissions_current",
        "BUILT_FORM": "built_form",
    })
    df = df.rename(columns={
        "electricity_avg_variable_per_kWh_price": "elec_price_kwh",
        "num_households_received_winter_fuel_payment_2023": "hh_winter_fuel_payment_2023",
        "avg_fuel_bill_annual_individual_estimate": "fuel_bill_individual_est",
        "local_authority_mean_domessic_electricity_consumption_kWh_per_household": "la_elec_use_kwh_hh",
        "local_authority_mean_domestic_gas_consumption_kWh_per_meter": "la_gas_use_kwh_meter",
        "avg_fuel_bill_annual_local_average_estimate": "fuel_bill_local_avg_est",
    })

    df["predicted_class"] = df["predicted_class"].astype("category")

    # Step 10: transforms, drops, caps and filters
    df["log_energy_per_room"] = np.log(df["energy_per_room"] + 1)
    df = df.drop(columns=["fuel_bill_difference"])
    df = df.drop(columns=["local_authority.x"])

    floor_cap = df["floor_area_per_room"].quantile(0.99)
    df = df[df["floor_area_per_room"] <= floor_cap].copy()

    fuel_ratio_cap = df["fuel_cost_ratio"].quantile(0.99)
    df = df[df["fuel_cost_ratio"] <= fuel_ratio_cap].copy()

    df["log_fuel_bill_individual_est"] = np.log(df["fuel_bill_individual_est"] + 1)

    for var in ["population_2024", "child_population", "senior_population", "working_age_population"]:
        cap = df[var].quantile(0.95)
        df = df[df[var] <= cap].copy()

    tas_cols_to_keep = ["tas_winter_1.5_median", "tas_winter_3.5_median"]
    tas_cols_to_drop = [
        c for c in df.columns
        if re.match(r"^tas_winter_.*_median$", c) and c not in tas_cols_to_keep
    ]
    df = df.drop(columns=tas_cols_to_drop)

    df = df.drop(columns=["postcode_district", "local_authority_code.x", "LMK_KEY"])
    df = df.drop(columns=["LODGEMENT_DATE"])

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype("category")
    df["predicted_class"] = df["predicted_class"].astype("category")

    # Step 10: dummy-encode predictors, reattach target
    target = df["predicted_class"]
    df_encoded = pd.get_dummies(df.drop(columns=["predicted_class"]), drop_first=True, dtype=int)
    df_ready = pd.concat([df_encoded, target.rename("predicted_class")], axis=1)

    # Step 10: SMOTE (all classes upsampled to the majority size, k=5) —
    # applied before the split, as in the notebook.
    X_smote = df_ready.drop(columns=["predicted_class"])
    y_smote = df_ready["predicted_class"]
    X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_smote, y_smote)
    df_balanced = X_bal.copy()
    df_balanced["predicted_class"] = y_bal

    # Step 08: stratified 80/20 split, seed 42
    train_data, test_data = train_test_split(
        df_balanced, test_size=0.2,
        stratify=df_balanced["predicted_class"], random_state=42,
    )
    train_data = train_data.copy()
    test_data = test_data.copy()
    train_data["predicted_class"] = train_data["predicted_class"].astype("category")
    test_data["predicted_class"] = test_data["predicted_class"].astype("category")

    # Step 11: feature sets — Pipeline A includes the EPC efficiency score;
    # Pipeline B removes it and re-adds the high-correlation proxies.
    label_map = list(train_data["predicted_class"].cat.categories)
    pipeline_a_vars = [c for c in train_data.columns if c != "predicted_class"]
    pipeline_b_vars = [c for c in pipeline_a_vars if c != "current_energy_efficiency"]
    pipeline_b_vars = list(dict.fromkeys(
        pipeline_b_vars + ["energy_consumption_current", "co2_per_m2", "energy_per_m2"]
    ))

    # XGBoost needs integer class labels; use the categorical codes for every
    # model so all reports share one consistent y. The category order comes
    # from the parquet dictionary (the R cut() ordering, Very Low..Very High),
    # matching the notebook's label_map.
    y_train = train_data["predicted_class"].cat.codes.to_numpy()
    y_test = test_data["predicted_class"].cat.codes.to_numpy()

    return {
        "n_raw": n_raw,
        "n_after_filters": len(df_ready),
        "n_balanced": len(df_balanced),
        "class_counts": y_bal.value_counts().to_dict(),
        "label_map": label_map,
        "train_data": train_data,
        "test_data": test_data,
        "pipeline_a_vars": pipeline_a_vars,
        "pipeline_b_vars": pipeline_b_vars,
        "y_train": y_train,
        "y_test": y_test,
    }


# --------------------------------------------------------------------------
# Model construction — hyperparameters mirror notebook Steps 11/12/14
# --------------------------------------------------------------------------
def make_xgb(n_classes: int) -> XGBClassifier:
    # Step 11: multi:softprob, mlogloss, depth 6, eta 0.1, subsample 0.8,
    # colsample 0.8, 150 rounds, early stopping 10, seed 42
    return XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=150,
        early_stopping_rounds=10,
        random_state=42,
    )


def make_rf() -> RandomForestClassifier:
    # Step 12: ntree=300, seed 42; mtry default = sqrt(p) ↔ max_features="sqrt".
    # n_jobs=-1 is a pure speed knob — statistically identical.
    return RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)


def make_logreg() -> LogisticRegression:
    # Step 14: nnet::multinom ↔ unregularised multinomial softmax, maxit 500
    return LogisticRegression(penalty=None, max_iter=500, solver="lbfgs")


def build_reports(prep: dict) -> dict[str, EstimatorReport]:
    train, test = prep["train_data"], prep["test_data"]
    y_train, y_test = prep["y_train"], prep["y_test"]
    n_classes = len(prep["label_map"])
    reports: dict[str, EstimatorReport] = {}

    for pipe, feats in (("A", prep["pipeline_a_vars"]), ("B", prep["pipeline_b_vars"])):
        X_train = train[feats].to_numpy(dtype=float)
        X_test = test[feats].to_numpy(dtype=float)

        # XGBoost: fitted externally so the notebook's early-stopping watchlist
        # (train, test) is preserved — skore's internal fit cannot pass eval_set.
        # The report is test-only as a result (no fit-time row for XGB).
        print(f"[{time.strftime('%H:%M:%S')}] fitting XGBoost {pipe} ...", flush=True)
        xgb = make_xgb(n_classes)
        xgb.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
        reports[f"XGBoost {pipe}"] = EstimatorReport(xgb, X_test=X_test, y_test=y_test)

        # RF / LogReg: unfitted — skore fits them (train metrics + timings captured)
        print(f"[{time.strftime('%H:%M:%S')}] fitting Random Forest {pipe} ...", flush=True)
        reports[f"Random Forest {pipe}"] = EstimatorReport(
            make_rf(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        )
        print(f"[{time.strftime('%H:%M:%S')}] fitting Logistic Regression {pipe} ...", flush=True)
        reports[f"Logistic Regression {pipe}"] = EstimatorReport(
            make_logreg(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        )

    return reports


# --------------------------------------------------------------------------
# Markdown assembly
# --------------------------------------------------------------------------
def write_markdown(prep: dict, reports: dict[str, EstimatorReport], elapsed: float) -> None:
    import skore
    import sklearn
    import xgboost
    import imblearn

    comparison = ComparisonReport(reports=reports)
    comp_frame = comparison.metrics.summarize().frame()

    legend = ", ".join(f"`{i}` = {lbl}" for i, lbl in enumerate(prep["label_map"]))
    class_counts = ", ".join(f"{k}: {v:,}" for k, v in sorted(prep["class_counts"].items()))

    lines: list[str] = []
    lines.append("# skore Evaluation — Fuel Poverty Risk Classifiers")
    lines.append("")
    lines.append(f"Generated by `skore_evaluation.py` (skore {skore.__version__}, "
                 f"scikit-learn {sklearn.__version__}, xgboost {xgboost.__version__}, "
                 f"imbalanced-learn {imblearn.__version__}). "
                 f"Total wall time: {elapsed/60:.1f} min.")
    lines.append("")
    lines.append("## Protocol")
    lines.append("")
    lines.append("- Data: `ml_model_data_with_predictions.parquet` "
                 f"({prep['n_raw']:,} rows) → Step 10 preprocessing "
                 f"({prep['n_after_filters']:,} rows) → SMOTE "
                 f"({prep['n_balanced']:,} rows; {class_counts})")
    lines.append("- Split: stratified 80/20, seed 42 (Step 08) — SMOTE is applied "
                 "before the split, mirroring the notebook pipeline exactly")
    lines.append("- Pipelines: **A** includes `current_energy_efficiency`; **B** excludes it "
                 "and re-adds `energy_consumption_current`, `co2_per_m2`, `energy_per_m2` (Step 11)")
    lines.append(f"- Class labels are the categorical codes: {legend}")
    lines.append("- XGBoost is fitted outside skore to preserve the notebook's early-stopping "
                 "watchlist, so its report is test-only (no fit-time row)")
    lines.append("- The DNN (Step 13) is excluded, matching the notebook's own Step 15 evaluation")
    lines.append("- Metrics are skore's multiclass defaults (no manual `scoring` overrides)")
    lines.append("")
    lines.append("## Model comparison (skore `ComparisonReport`)")
    lines.append("")
    lines.append("```text")
    lines.append(comp_frame.to_string())
    lines.append("```")
    lines.append("")
    lines.append("## Per-model reports (skore `EstimatorReport.to_markdown()`)")
    for name, rep in reports.items():
        lines.append("")
        lines.append(f"### {name}")
        lines.append("")
        lines.append(rep.to_markdown())

    OUTPUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUTPUT_MD}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="Path to ml_model_data_with_predictions.parquet")
    args = parser.parse_args()

    if not args.data.exists():
        sys.exit(f"Data file not found: {args.data}")

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] preparing data from {args.data} ...", flush=True)
    prep = prepare_data(args.data)
    print(f"    raw {prep['n_raw']:,} → filtered {prep['n_after_filters']:,} "
          f"→ SMOTE-balanced {prep['n_balanced']:,} rows", flush=True)

    reports = build_reports(prep)

    print(f"[{time.strftime('%H:%M:%S')}] building comparison + markdown ...", flush=True)
    write_markdown(prep, reports, time.time() - t0)


if __name__ == "__main__":
    main()
