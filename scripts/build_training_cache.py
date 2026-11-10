#!/usr/bin/env python3
"""Build the cached inputs for the local dB/dt forecast model.

For every year in YEARS (env, space separated):
  analysis/omni_hourly_<year>.json        hourly solar-wind features from NASA OMNI 1-minute data
  analysis/ground_hourly_<STA>_<year>.json hourly max dB/dt for extra stations (default: THY, Tihany)
Existing files are kept, so reruns only download what is missing.
"""
import csv
import io
import json
import os
import sys
import time
import urllib.parse
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from false_alarm_study import HAPI, MAX_PLAUSIBLE, http  # noqa: E402
from fetch_live_magnetometers import dbdt, fetch_gin  # noqa: E402

YEARS = [int(y) for y in os.environ.get("YEARS", "2024").split()]
EXTRA = os.environ.get("EXTRA_STATIONS", "THY").split()
PARAMS = ["BY_GSM", "BZ_GSM", "flow_speed", "proton_density", "Pressure"]
LIMIT = {"BY_GSM": 999, "BZ_GSM": 999, "flow_speed": 9999, "proton_density": 900, "Pressure": 99}


def omni_month(a, b, params):
    q = urllib.parse.urlencode({"id": "OMNI_HRO2_1MIN", "parameters": ",".join(params), "format": "csv",
                                "include": "header",
                                "time.min": a.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "time.max": b.strftime("%Y-%m-%dT%H:%M:%SZ")})
    text = http(f"{HAPI}/data?{q}")
    head = "\n".join(l[1:] for l in text.splitlines() if l.startswith("#"))
    names = [p["name"] for p in json.loads(head)["parameters"]]  # dataset order, Time first
    rows = []
    for r in csv.reader(io.StringIO("\n".join(l for l in text.splitlines() if not l.startswith("#")))):
        try:
            t = int(datetime.strptime(r[0][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
            vals = {n: float(x) for n, x in zip(names[1:], r[1:])}
        except (ValueError, IndexError):
            continue
        rows.append((t, {n: (v if abs(v) < LIMIT.get(n, 1e9) else None) for n, v in vals.items()}))
    return rows


def omni_hourly(year):
    out = f"analysis/omni_hourly_{year}.json"
    if os.path.exists(out):
        print(f"{out} exists")
        return
    hours = {}
    for m in range(1, 13):
        a = datetime(year, m, 1, tzinfo=timezone.utc)
        b = datetime(year + (m == 12), m % 12 + 1, 1, tzinfo=timezone.utc)
        rows = None
        for attempt in range(3):
            try:
                rows = omni_month(a, b, PARAMS)
                break
            except Exception as e:  # noqa: BLE001
                print(f"OMNI {a:%Y-%m} attempt {attempt + 1}: {str(e)[:150]}")
                time.sleep(10)
        if rows is None:
            continue
        for t, v in rows:
            hours.setdefault(t - t % 3600, []).append(v)
        print(f"OMNI {a:%Y-%m}: {len(rows)} rows")
    res = {}
    for h, mins in hours.items():
        bz = [m["BZ_GSM"] for m in mins if m.get("BZ_GSM") is not None]
        by = [m["BY_GSM"] for m in mins if m.get("BY_GSM") is not None]
        v = [m["flow_speed"] for m in mins if m.get("flow_speed") is not None]
        n = [m["proton_density"] for m in mins if m.get("proton_density") is not None]
        p = [m["Pressure"] for m in mins if m.get("Pressure") is not None]
        ey = [m["flow_speed"] * max(0.0, -m["BZ_GSM"]) * 1e-3 for m in mins
              if m.get("BZ_GSM") is not None and m.get("flow_speed") is not None]
        if not bz:
            continue
        mean = lambda x: round(sum(x) / len(x), 2) if x else None  # noqa: E731
        res[h] = {"n": len(ey), "bz_mean": mean(bz), "bz_min": round(min(bz), 2), "bz_std": round(
            (sum((x - sum(bz) / len(bz)) ** 2 for x in bz) / len(bz)) ** 0.5, 2),
            "by_mean": mean(by), "v_mean": mean(v), "v_max": max(v) if v else None,
            "n_mean": mean(n), "p_mean": mean(p), "p_max": max(p) if p else None,
            "ey_mean": mean(ey), "ey_max": round(max(ey), 2) if ey else None,
            "ey5_min": sum(1 for e in ey if e >= 5)}
    if len(res) < 1000:
        print(f"OMNI {year}: only {len(res)} hours, not cached")
        return
    with open(out, "w") as f:
        json.dump(res, f, separators=(",", ":"))
    print(f"{out}: {len(res)} hours")


def ground_station(iaga, year):
    out = f"analysis/ground_hourly_{iaga}_{year}.json"
    if os.path.exists(out):
        print(f"{out} exists")
        return
    hourly, got = {}, 0
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    t = start
    while t < end:
        u = min(t + timedelta(days=10), end)
        pts = None
        for attempt in range(3):
            try:
                pts = fetch_gin(iaga, t, u - timedelta(minutes=1))
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    print(f"{iaga} {t:%Y-%m-%d}: {str(e)[:120]}")
                time.sleep(5)
        if pts:
            for ts, v in dbdt([p for p in pts if t.timestamp() <= p[0] < u.timestamp()]):
                if v > MAX_PLAUSIBLE:
                    continue
                h = ts - ts % 3600
                if v > hourly.get(h, 0):
                    hourly[h] = round(v, 1)
                got += 1
        t = u
    cover = round(got / ((end - start).total_seconds() / 60), 3)
    print(f"{iaga} {year}: coverage {cover:.0%}")
    if cover < 0.2:
        print(f"{iaga} {year}: too little data, not cached")
        return
    with open(out, "w") as f:
        json.dump({"hourly": hourly, "coverage": cover}, f, separators=(",", ":"))


def main():
    os.makedirs("analysis", exist_ok=True)
    for y in YEARS:
        print(f"##### OMNI {y}")
        omni_hourly(y)
    for y in YEARS:
        for s in EXTRA:
            ground_station(s, y)
    return 0


if __name__ == "__main__":
    sys.exit(main())
