#!/usr/bin/env python3
"""Fetch recent SuperMAG magnetometer data and compute dB/dt per station -> supermag.json.

Needs env SM_LOGON = the SuperMAG *username* (the login name, not the email).

supermag.json format:
  {"fetched_at": ISO, "data_time": ISO of newest sample, "stations": [
     {"id", "name", "lat", "lng", "dbdt": peak nT/min over the last hour of data,
      "dbdt_now": latest nT/min, "time": ISO of latest sample}], "errors": [...]}

dB/dt is the horizontal rate of change: sqrt(dN^2 + dE^2) per minute between consecutive
1-minute samples. That rate, not the size of the disturbance, is what drives GIC.
If nothing usable comes back, the previous stations are kept and marked stale.
"""
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

API = "https://supermag.jhuapl.edu/services/data-api.php"
WINDOW_HOURS = 6  # SuperMAG near-real-time data can lag; ask for a wide window
PEAK_MINUTES = 60

# IAGA code: (name, geographic lat, lon)
STATIONS = {
    "KIR": ("Kiruna, SE", 67.84, 20.42),
    "ESK": ("Eskdalemuir, UK", 55.31, -3.21),
    "HAD": ("Hartland, UK", 50.98, -4.48),
    "OTT": ("Ottawa, CA", 45.40, -75.55),
    "FRD": ("Fredericksburg, US", 38.20, -77.37),
    "BRW": ("Barrow, US", 71.32, -156.62),
    "HER": ("Hermanus, ZA", -34.43, 19.23),
}


def comp(rec, key):
    """Field value of N or E; accepts {"N": {"nez": x}} or flat keys."""
    v = rec.get(key)
    if isinstance(v, dict):
        v = v.get("nez", v.get("geo"))
    if v is None:
        v = rec.get(key.lower() + "_nez", rec.get("db" + key.lower() + "_nez"))
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if abs(f) >= 99999 else f  # 999999 = missing in SuperMAG


def fetch_station(logon, iaga, start, extent):
    q = urllib.parse.urlencode({
        "fmt": "json", "logon": logon, "start": start,
        "extent": f"{extent:012d}", "station": iaga,
    })
    req = urllib.request.Request(f"{API}?{q}", headers={"User-Agent": "helios-shield-demo"})
    with urllib.request.urlopen(req, timeout=40) as r:
        text = r.read().decode("utf-8", "replace").strip()
    if not text or text.upper().startswith("ERROR"):
        raise RuntimeError(text[:200] or "empty response")
    data = json.loads(text)
    if not isinstance(data, list):
        raise RuntimeError(f"unexpected response: {str(data)[:200]}")
    return data


def dbdt_series(records):
    pts = []
    for r in records:
        try:
            t = int(float(r.get("tval")))
        except (TypeError, ValueError):
            continue
        n, e = comp(r, "N"), comp(r, "E")
        if n is not None and e is not None:
            pts.append((t, n, e))
    pts.sort()
    out = []
    for (t0, n0, e0), (t1, n1, e1) in zip(pts, pts[1:]):
        dt_min = (t1 - t0) / 60.0
        if 0 < dt_min <= 2:  # skip gaps
            out.append((t1, math.hypot(n1 - n0, e1 - e0) / dt_min))
    return out


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    logon = os.environ.get("SM_LOGON", "").strip()
    now = datetime.now(timezone.utc)
    result = {"fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "data_time": None, "stations": [], "errors": []}
    if not logon:
        result["errors"].append("SM_LOGON is not set")

    start = (now - timedelta(hours=WINDOW_HOURS)).strftime("%Y-%m-%dT%H:%M")
    extent = WINDOW_HOURS * 3600
    newest = 0
    for iaga, (name, lat, lng) in STATIONS.items() if logon else []:
        try:
            series = dbdt_series(fetch_station(logon, iaga, start, extent))
        except Exception as e:  # noqa: BLE001
            result["errors"].append(f"{iaga}: {e}")
            print(f"{iaga}: failed: {e}")
            continue
        if not series:
            result["errors"].append(f"{iaga}: no samples in window")
            print(f"{iaga}: no samples")
            continue
        last_t = series[-1][0]
        recent = [v for t, v in series if t > last_t - PEAK_MINUTES * 60]
        newest = max(newest, last_t)
        result["stations"].append({
            "id": iaga, "name": name, "lat": lat, "lng": lng,
            "dbdt": round(max(recent), 1), "dbdt_now": round(series[-1][1], 1), "time": iso(last_t),
        })
        print(f"{iaga}: {len(series)} samples, peak {max(recent):.1f} nT/min, latest {iso(last_t)}")

    if newest:
        result["data_time"] = iso(newest)

    if not result["stations"]:
        try:
            with open("supermag.json") as f:
                prev = json.load(f)
            if prev.get("stations"):
                result["stations"] = prev["stations"]
                result["data_time"] = prev.get("data_time")
                result["stale"] = True
                print("no new data, keeping previous stations (stale)")
        except Exception:  # noqa: BLE001
            pass

    with open("supermag.json.tmp", "w") as f:
        json.dump(result, f, separators=(",", ":"))
    os.replace("supermag.json.tmp", "supermag.json")
    print(json.dumps(result)[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
