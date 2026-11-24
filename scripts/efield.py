#!/usr/bin/env python3
"""Surface geoelectric field from magnetometer data and a layered (1-D) ground model.

Method (plane-wave, as used in GIC studies): for each frequency the ground's surface impedance
Z(w) follows from the layer stack by the standard recursion; then
    Ex(w) =  Z(w) * By(w) / mu0      (northward E from eastward B)
    Ey(w) = -Z(w) * Bx(w) / mu0      (eastward E from northward B)
The magnetic series is detrended, tapered and zero-padded before the FFT.

Ground models: Pannonian Basin layer sequences No 1 and No 3 from Table 1 of
  Adam A., Szarka L., Novak A., Wesztergom V.: "Key results on deep electrical conductivity
  anomalies in the Pannonian Basin (PB), and their geodynamic aspects" (real.mtak.hu/73548).
Two models are kept to show how much the answer depends on the ground: a 1-D model is a
regional approximation, not a site survey.

Modes:
  live (default):   last 48 h of Tihany (THY) minute data -> efield.json (last 24 h kept)
  REPLAY=YYYY-MM-DD:DAYS  e.g. 2024-05-10:2 -> storms/efield_<date>_THY.json
"""
import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_live_magnetometers import despike, fetch_gin  # noqa: E402

MU0 = 4e-7 * math.pi
STATION = os.environ.get("EF_STATION", "THY")
SOURCE = "Adam, Szarka, Novak, Wesztergom: Key results on deep electrical conductivity anomalies " \
         "in the Pannonian Basin, Table 1 (real.mtak.hu/73548)"
# (thickness km, resistivity ohm-m) top down; last entry is the half-space resistivity.
MODELS = {
    "pb1": {"label": "Pannonian Basin model No 1 (5 ohm-m sediment over 300 ohm-m crust)",
            "layers": [(0.50, 5.0), (10.0, 300.0), (0.12, 0.10), (40.0, 60.0), (15.0, 15.0)], "half": 5.0},
    "pb3": {"label": "Pannonian Basin model No 3 (1.7 ohm-m sediment over 70 ohm-m crust)",
            "layers": [(1.20, 1.70), (14.0, 70.0), (0.70, 1.00), (25.0, 50.0), (25.0, 15.0)], "half": 5.0},
}


def impedance(freq_hz, model):
    """Surface impedance Z (ohm) of a layered half-space, plane-wave source."""
    w = 2 * np.pi * np.asarray(freq_hz, dtype=float)
    w = np.where(w == 0, 1e-12, w)
    k = np.sqrt(1j * w * MU0 / model["half"])
    z = 1j * w * MU0 / k
    for d_km, rho in reversed(model["layers"]):
        kj = np.sqrt(1j * w * MU0 / rho)
        eta = 1j * w * MU0 / kj
        t = np.tanh(kj * d_km * 1000.0)
        z = eta * (z + eta * t) / (eta + z * t)
    return z


def regular(pts, max_gap_min=15):
    """Minute grid from sorted (t, x, y); short gaps linearly filled, long gaps flagged."""
    t0, t1 = pts[0][0], pts[-1][0]
    grid = np.arange(t0, t1 + 1, 60)
    ts = np.array([p[0] for p in pts], dtype=float)
    x = np.interp(grid, ts, [p[1] for p in pts])
    y = np.interp(grid, ts, [p[2] for p in pts])
    gap = np.zeros(len(grid), dtype=bool)
    for a, b in zip(ts, ts[1:]):
        if b - a > max_gap_min * 60:
            gap[(grid > a) & (grid < b)] = True
    return grid, x, y, gap


def efield(x_nt, y_nt, model):
    """E north/east in mV/km for a regular 1-minute series of X (north) and Y (east) in nT."""
    n = len(x_nt)
    idx = np.arange(n)
    out = []
    for comp in (x_nt, y_nt):
        c = np.asarray(comp, dtype=float)
        c = c - np.polyval(np.polyfit(idx, c, 1), idx)       # remove trend
        taper = np.ones(n)
        m = max(1, n // 20)                                   # 5 % cosine taper at both ends
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        taper[:m], taper[-m:] = ramp, ramp[::-1]
        out.append(c * taper)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    f = np.fft.rfftfreq(nfft, d=60.0)
    z = impedance(f, model)
    bx = np.fft.rfft(out[0], nfft) * 1e-9
    by = np.fft.rfft(out[1], nfft) * 1e-9
    e_n = np.fft.irfft(z * by / MU0, nfft)[:n] * 1e6          # V/m -> mV/km
    e_e = np.fft.irfft(-z * bx / MU0, nfft)[:n] * 1e6
    return e_n, e_e


def compute(pts, keep_from):
    pts, removed = despike(sorted(pts))
    grid, x, y, gap = regular(pts)
    res = {"station": STATION, "source": SOURCE, "models": {}, "despiked": removed}
    keep = grid >= keep_from
    res["t0"] = int(grid[keep][0])
    res["step_s"] = 60
    res["gap"] = [int(i) for i in np.nonzero(gap[keep])[0]]
    for key, model in MODELS.items():
        e_n, e_e = efield(x, y, model)
        en, ee = e_n[keep], e_e[keep]
        mag = np.hypot(en, ee)
        mag[gap[keep]] = 0
        res["models"][key] = {
            "label": model["label"],
            "en": [round(float(v), 1) for v in en],
            "ee": [round(float(v), 1) for v in ee],
            "peak": round(float(mag.max()), 1),
            "peak_t": int(grid[keep][int(mag.argmax())]),
            "now": round(float(mag[-1]), 1),
        }
    return res


def fetch(start, end):
    pts = []
    t = start
    while t < end:  # GIN in chunks of up to 5 days
        u = min(t + timedelta(days=5), end)
        pts += fetch_gin(STATION, t, u - timedelta(minutes=1))
        t = u
    return sorted({p[0]: p for p in pts if start.timestamp() <= p[0] < end.timestamp()}.values())


def main():
    replay = os.environ.get("REPLAY")
    if replay:
        day, days = replay.split(":")
        start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = start + timedelta(days=int(days))
        pts = fetch(start - timedelta(hours=12), end + timedelta(hours=6))  # spin-up margins
        res = compute(pts, start.timestamp())
        n_keep = int((end - start).total_seconds() // 60)
        for m in res["models"].values():
            m["en"], m["ee"] = m["en"][:n_keep], m["ee"][:n_keep]
        res["gap"] = [i for i in res["gap"] if i < n_keep]
        out = f"storms/efield_{day}_{STATION}.json"
    else:
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        pts = fetch(now - timedelta(hours=48), now + timedelta(minutes=1))
        if not pts or now.timestamp() - pts[-1][0] > 3 * 3600:
            print("no recent data, keeping previous efield.json")
            return 0
        res = compute(pts, pts[-1][0] - 24 * 3600 + 60)
        res["fetched_at"] = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        out = "efield.json"
    with open(out + ".tmp", "w") as f:
        json.dump(res, f, separators=(",", ":"))
    os.replace(out + ".tmp", out)
    for k, m in res["models"].items():
        print(f"{out} {k}: peak {m['peak']} mV/km at "
              f"{datetime.fromtimestamp(m['peak_t'], timezone.utc):%Y-%m-%d %H:%M}, now {m['now']} mV/km")
    return 0


if __name__ == "__main__":
    sys.exit(main())
