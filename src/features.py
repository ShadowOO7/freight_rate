"""Cleaning + feature engineering shared by experiments and final training."""
from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHT_CAP = 47_500.0


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data-quality issues. Works identically on train and validation."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    # 1) negative weights are sign errors (magnitudes look like normal loads) -> abs()
    out["weight_was_negative"] = (out["weight"] < 0).astype(int)
    out["weight"] = out["weight"].abs()
    # 2) weight is capped at 47,500; flag it so the model knows it is censored
    out["weight_at_cap"] = (out["weight"] >= WEIGHT_CAP).astype(int)
    # 3) flag missing values (filled later by Imputer)
    out["weight_missing"] = out["weight"].isna().astype(int)
    out["market_missing"] = out["market_index"].isna().astype(int)
    return out


class Imputer:
    """Group medians learned on training data only (no leakage)."""

    def fit(self, df: pd.DataFrame) -> "Imputer":
        self.weight_by_equip = df.groupby("equipment")["weight"].median()
        self.market_median = df["market_index"].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["weight"] = out["weight"].fillna(out["equipment"].map(self.weight_by_equip))
        day_mean = out.groupby("date")["market_index"].transform("mean")
        out["market_index"] = out["market_index"].fillna(day_mean).fillna(self.market_median)
        return out


def day_table(*frames: pd.DataFrame) -> pd.DataFrame:
    """Per-date market context from FEATURES only (no labels), pooled over every file we have."""
    all_ = pd.concat([f[["date", "market_index", "quote_signal"]] for f in frames])
    day = all_.groupby("date").agg(mi_day=("market_index", "mean"), qs_day=("quote_signal", "mean"))
    day = day.asfreq("D").interpolate(limit_direction="both")
    day["mi_roll7"] = day["mi_day"].rolling(7, center=True, min_periods=1).mean()
    day["mi_roll3"] = day["mi_day"].rolling(3, center=True, min_periods=1).mean()
    day["mi_lag7"] = day["mi_day"].shift(7).bfill()
    day["mi_lead7"] = day["mi_day"].shift(-7).ffill()
    return day


def build_features(df: pd.DataFrame, day: pd.DataFrame | None = None) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    dist = df["distance"].astype(float)
    hav = haversine_miles(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    X["distance"] = dist
    X["log_distance"] = np.log(dist)
    X["inv_distance"] = 1.0 / dist  # short hauls have higher rate/mile
    X["haversine"] = hav
    X["dist_ratio"] = dist / hav.clip(lower=1)
    X["weight"] = df["weight"]
    X["weight_at_cap"] = df["weight_at_cap"]
    X["weight_missing"] = df["weight_missing"]
    X["weight_was_negative"] = df["weight_was_negative"]
    X["market_index"] = df["market_index"]
    X["market_missing"] = df["market_missing"]
    X["quote_signal"] = df["quote_signal"]
    X["pickup_lat"] = df.pickup_lat
    X["pickup_lon"] = df.pickup_lon
    X["delivery_lat"] = df.delivery_lat
    X["delivery_lon"] = df.delivery_lon
    X["quote_total"] = df["quote_signal"] * dist
    X["log_quote_total"] = np.log(X["quote_total"])
    X["qs_dev"] = (df["quote_signal"] - 2.06).abs()
    X["weight_sat"] = df["weight"].clip(upper=44_000)
    X["dow"] = df["date"].dt.dayofweek
    if day is not None:
        d = day.reindex(df["date"]).reset_index(drop=True)
        d.index = df.index
        for c in day.columns:
            X[c] = d[c]
        X["mi_dev"] = X["market_index"] - X["mi_day"]
    for eq in ["Dry Van", "Reefer", "Flatbed"]:
        X[f"eq_{eq.replace(' ', '_')}"] = (df["equipment"] == eq).astype(int)
    return X