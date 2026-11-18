"""End-to-end pipeline: validate (time-ordered CV) -> fit final model -> predict -> write outputs.

Run from the repo root:  python src/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).parent))
from features import Imputer, build_features, clean, day_table  # noqa: E402

DATA = Path("data")
OUT = Path("results")
OUT.mkdir(exist_ok=True)
SEED = 0


def _hgb(loss: str, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss, max_iter=500, learning_rate=0.05, min_samples_leaf=40,
        l2_regularization=1.0 if loss == "squared_error" else 0.0,
        early_stopping=False, random_state=seed,
    )


def trim_mask(X: pd.DataFrame, y_log: np.ndarray, k: float = 4.0) -> np.ndarray:
    """True = keep. Drops rows whose log-rate residual is > k robust sigmas (corrupted targets)."""
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=SEED).fit(X, y_log)
    r = y_log - m.predict(X)
    mad = np.median(np.abs(r - np.median(r))) * 1.4826
    return np.abs(r - np.median(r)) <= k * mad


class FreightModel:
    """Log-rate ensemble: squared-error HGB on trimmed data + absolute-error HGB (robust)."""

    def fit(self, X: pd.DataFrame, rate: np.ndarray) -> "FreightModel":
        y = np.log(rate)
        keep = trim_mask(X, y)
        self.n_trimmed = int((~keep).sum())
        self.m_sq = _hgb("squared_error", SEED).fit(X[keep], y[keep])
        self.m_abs = _hgb("absolute_error", SEED + 1).fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(0.5 * self.m_sq.predict(X) + 0.5 * self.m_abs.predict(X))


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {"MAE": float(np.mean(np.abs(y - p))), "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "MAPE_%": float(np.mean(np.abs(y - p) / y) * 100), "MedAPE_%": float(np.median(np.abs(y - p) / y) * 100)}


def main() -> None:
    train_raw = pd.read_csv(DATA / "train_test.csv")
    val_raw = pd.read_csv(DATA / "validation.csv")
    template = pd.read_csv(DATA / "validation_predictions_template.csv")

    train, val = clean(train_raw), clean(val_raw)
    day = day_table(train, val)  # features only, no labels
    train["month"] = train["date"].dt.month

    # 1) Time-ordered validation: always train on the past, score on the future.
    folds = [([1, 2, 3, 4, 5, 6], [7, 8]), ([1, 2, 3, 4, 5, 6, 7, 8], [9, 10]), ([1, 2, 3, 4, 5, 6, 7], [8, 9])]
    cv = []
    for tr_m, va_m in folds:
        tr, va = train[train.month.isin(tr_m)], train[train.month.isin(va_m)]
        imp = Imputer().fit(tr)
        tr, va = imp.transform(tr), imp.transform(va)
        Xtr, Xva = build_features(tr, day), build_features(va, day)
        model = FreightModel().fit(Xtr, tr.posted_rate.values)
        p = model.predict(Xva)
        yv = va.posted_rate.values
        keep = trim_mask(Xva, np.log(yv))  # "clean" view: ignore corrupted holdout targets
        cv.append({"train_months": f"1-{tr_m[-1]}", "val_months": str(va_m),
                   "all": metrics(yv, p), "clean": metrics(yv[keep], p[keep])})
        print(cv[-1])
    (OUT / "cv_metrics.json").write_text(json.dumps(cv, indent=2))

    # 2) Final fit on ALL labeled data
    imp = Imputer().fit(train)
    train_i, val_i = imp.transform(train), imp.transform(val)
    model = FreightModel().fit(build_features(train_i, day), train_i.posted_rate.values)
    print(f"final model: trimmed {model.n_trimmed} suspect-target rows of {len(train_i)}")

    # 3) Predict validation.csv and fill the template
    pred = pd.Series(model.predict(build_features(val_i, day)), index=val_i["load_id"])
    out = template[["load_id"]].copy()
    out["predicted_rate"] = out["load_id"].map(pred).round(2)
    assert out["predicted_rate"].notna().all() and (out["predicted_rate"] > 0).all() and len(out) == 12_000
    out.to_csv("validation_predictions.csv", index=False)
    print("wrote validation_predictions.csv", out.shape)

    # 4) Fixed December chart inputs: only the date changes
    coords = pd.concat([
        train_raw[["pickup", "pickup_lat", "pickup_lon"]].rename(columns=lambda c: c.replace("pickup", "city")),
        train_raw[["delivery", "delivery_lat", "delivery_lon"]].rename(columns=lambda c: c.replace("delivery", "city")),
    ]).drop_duplicates("city").set_index("city")
    dates = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    dec = pd.DataFrame({"pickup": "Lexington", "delivery": "Fort Wayne", "distance": 360.0,
                        "equipment": "Dry Van", "weight": 32000.0, "date": dates})
    dec_feat = dec.copy()
    for side in ("pickup", "delivery"):
        dec_feat[f"{side}_lat"] = coords.loc[dec[side], "city_lat"].values
        dec_feat[f"{side}_lon"] = coords.loc[dec[side], "city_lon"].values
    dec_feat["market_index"] = day.reindex(dates)["mi_day"].values
    dec_feat["quote_signal"] = day.reindex(dates)["qs_day"].values
    for c in ("weight_was_negative", "weight_at_cap", "weight_missing", "market_missing"):
        dec_feat[c] = 0
    dec["predicted_rate"] = model.predict(build_features(dec_feat, day)).round(2)
    dec.assign(date=dec["date"].dt.strftime("%Y-%m-%d")).to_csv(DATA / "december_chart_inputs.csv", index=False)
    print("wrote data/december_chart_inputs.csv", dec.shape)


if __name__ == "__main__":
    main()