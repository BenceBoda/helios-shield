#!/usr/bin/env python3
"""Local dB/dt forecast: can a small ML model beat the plain Ey rule, station by station?

Question per station and hour h: will the hourly max dB/dt reach THR nT/min?
Inputs are only what is known before hour h starts:
  - hourly solar-wind features from OMNI for hours h-1 .. h-6   (model "sw")
  - plus the station's own dB/dt in hours h-1, h-2               (model "sw+ground")
  "sw_lin" is a plain logistic regression on the solar-wind features, as a sturdier check.
Baselines scored on the same hours:
  - ey_max:      strongest Ey in hour h-1 (the quantity the V1 rule watches)
  - persistence: the station's own dB/dt in hour h-1
Evaluation is leave-one-year-out: each year is predicted by a model trained on the other
years. Reported per station and threshold: positives, average precision (AP; chance level =
share of positive hours) and recall at precision >= 0.5 (hit rate when half the alerts are right).
Hours whose high value no second station confirms are dropped as instrument artefacts.
Hours with under 30 valid OMNI minutes in h-1 are skipped for every method ("blind" hours).
Output: analysis/local_forecast_eval.json
"""
import glob
import json
import math
import os
import re
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THRESHOLDS = [30.0, 100.0, 300.0]
MIN_POS = 12
REGIONS = [{"FRD", "BOU", "NEW", "SIT"}, {"NGK", "ESK", "HAD", "LER", "THY"}]
KEYS = ["bz_mean", "bz_min", "bz_std", "by_mean", "v_mean", "v_max", "n_mean", "p_mean", "p_max",
        "ey_mean", "ey_max", "ey5_min"]
NAN = float("nan")


def load():
    omni, ground = {}, {}
    for f in sorted(glob.glob("analysis/omni_hourly_*.json")):
        y = int(re.search(r"(\d{4})", f).group(1))
        omni[y] = {int(h): v for h, v in json.load(open(f)).items()}
    for y in omni:
        g = {}
        f = f"analysis/ground_hourly_{y}.json"
        if os.path.exists(f):
            for h, row in json.load(open(f))["hourly"].items():
                g[int(h)] = dict(row)
        for f in glob.glob(f"analysis/ground_hourly_*_{y}.json"):
            sta = re.search(r"ground_hourly_([A-Z]{3})_", f).group(1)
            for h, v in json.load(open(f))["hourly"].items():
                g.setdefault(int(h), {})[sta] = v
        ground[y] = g
    return omni, ground


def feat(o, h):
    """Solar-wind features from hours h-1..h-6."""
    x = []
    lag = [o.get(h - k * 3600) for k in range(1, 7)]
    for k in range(3):
        x += [(lag[k] or {}).get(key, NAN) if (lag[k] or {}).get(key) is not None else NAN for key in KEYS]
    ey6 = [l["ey_max"] for l in lag if l and l.get("ey_max") is not None]
    bz6 = [l["bz_min"] for l in lag if l]
    eym = [l["ey_mean"] for l in lag if l and l.get("ey_mean") is not None]
    x += [max(ey6) if ey6 else NAN, sum(eym) if eym else NAN, min(bz6) if bz6 else NAN]
    p1 = (lag[0] or {}).get("p_max"); p2 = (lag[1] or {}).get("p_mean")
    x.append(p1 - p2 if p1 is not None and p2 is not None else NAN)
    # Time of day matters physically (local time of the station). Day of year is left out on
    # purpose: with a handful of storms per year a model just memorises their dates.
    ut = (h % 86400) / 86400 * 2 * math.pi
    x += [math.sin(ut), math.cos(ut)]
    return x


def confirmed(row, sta, thr):
    """False if sta reads >= thr but no other station in its region reads >= thr/3."""
    region = next((r for r in REGIONS if sta in r), set())
    others = [v for s, v in row.items() if s != sta and s in region]
    return bool(others) and max(others) >= thr / 3


def dataset(omni, ground, sta):
    rows = []
    for y in sorted(omni):
        o, g = omni[y], ground[y]
        for h in sorted(g):
            v = g[h].get(sta)
            last = o.get(h - 3600)
            if v is None or not last or last["n"] < 30:
                continue
            if v >= THRESHOLDS[0] and not confirmed(g[h], sta, v):
                continue
            g1 = g.get(h - 3600, {}).get(sta, NAN)
            g2 = g.get(h - 7200, {}).get(sta, NAN)
            rows.append((y, v, feat(o, h), [g1, g2], last.get("ey_max") or 0.0))
    return rows


def recall_at_precision(y, s, p_min=0.5):
    p, r, _ = precision_recall_curve(y, s)
    ok = r[p >= p_min]
    return float(ok[:-1].max()) if len(ok) > 1 else 0.0  # drop the trivial (p=1, r=0) end point


def main():
    omni, ground = load()
    years = sorted(omni)
    stations = sorted({s for g in ground.values() for row in g.values() for s in row})
    print("years", years, "stations", stations)
    out = {"years": years, "definitions": __doc__.strip(), "results": []}
    print(f"\n{'sta':4} {'thr':>4} {'pos':>5} {'hours':>7} {'chance':>7} | "
          f"{'AP ey':>6} {'AP pers':>7} {'AP lin':>6} {'AP sw':>6} {'AP sw+g':>7} | "
          f"{'R ey':>5} {'R pers':>6} {'R lin':>5} {'R sw':>5} {'R sw+g':>6}")
    for sta in stations:
        rows = dataset(omni, ground, sta)
        if len(rows) < 5000:
            continue
        yr = np.array([r[0] for r in rows])
        val = np.array([r[1] for r in rows])
        Xs = np.array([r[2] for r in rows], dtype=float)
        Xg = np.hstack([Xs, np.array([r[3] for r in rows], dtype=float)])
        ey = np.array([r[4] for r in rows])
        pers = np.nan_to_num(np.array([r[3][0] for r in rows], dtype=float))
        for thr in THRESHOLDS:
            y = (val >= thr).astype(int)
            if y.sum() < MIN_POS:
                continue
            pred = {"sw_lin": np.zeros(len(y)), "sw": np.zeros(len(y)), "sw+ground": np.zeros(len(y))}
            Xl = np.nan_to_num(Xs)
            for test in years:
                tr, te = yr != test, yr == test
                if te.sum() == 0 or y[tr].sum() < 5:
                    continue
                lin = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000,
                                                                         class_weight="balanced"))
                lin.fit(Xl[tr], y[tr])
                pred["sw_lin"][te] = lin.predict_proba(Xl[te])[:, 1]
                for name, X in (("sw", Xs), ("sw+ground", Xg)):
                    m = HistGradientBoostingClassifier(max_iter=120, learning_rate=0.05, max_depth=3,
                                                       min_samples_leaf=10, l2_regularization=1.0,
                                                       random_state=0)
                    m.fit(X[tr], y[tr])
                    pred[name][te] = m.predict_proba(X[te])[:, 1]
            scores = {"ey_max": ey, "persistence": pers, **pred}
            res = {"station": sta, "thr": thr, "positives": int(y.sum()), "hours": int(len(y)),
                   "chance": round(float(y.mean()), 5),
                   "ap": {k: round(float(average_precision_score(y, s)), 3) for k, s in scores.items()},
                   "recall_at_p50": {k: round(recall_at_precision(y, s), 3) for k, s in scores.items()},
                   "positives_by_year": {int(t): int(y[yr == t].sum()) for t in years}}
            out["results"].append(res)
            a, r = res["ap"], res["recall_at_p50"]
            print(f"{sta:4} {thr:4.0f} {res['positives']:5d} {res['hours']:7d} {res['chance']:7.4f} | "
                  f"{a['ey_max']:6.3f} {a['persistence']:7.3f} {a['sw_lin']:6.3f} {a['sw']:6.3f} {a['sw+ground']:7.3f} | "
                  f"{r['ey_max']:5.2f} {r['persistence']:6.2f} {r['sw_lin']:5.2f} {r['sw']:5.2f} {r['sw+ground']:6.2f}")
    with open("analysis/local_forecast_eval.json", "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
