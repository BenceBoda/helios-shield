#!/usr/bin/env python3
"""Fetch near-real-time 1-minute magnetometer data and compute dB/dt -> live_stations.json.

Sources (both public, no login):
  - USGS Geomagnetism web service (US observatories)
  - INTERMAGNET via the BGS GIN web service (European and other observatories)

live_stations.json format:
  {"fetched_at": ISO, "data_time": ISO of newest sample, "stations": [
     {"id", "name", "source", "lat", "lng", "dbdt": peak nT/min over the last hour,
      "dbdt_now": latest nT/min, "time": ISO of latest sample, "lag_min": minutes behind now}],
   "errors": [...]}

dB/dt = sqrt(dX^2 + dY^2) per minute between consecutive 1-minute samples.
A station is dropped if its newest sample is older than MAX_LAG_MIN.
If nothing usable comes back, the previous stations are kept and marked stale.
"""
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

WINDOW_HOURS = 3
PEAK_MINUTES = 60
MAX_LAG_MIN = 180
OUT = "live_stations.json"
UA = {"User-Agent": "helios-shield-demo"}

USGS = {  # IAGA: (name, lat, lon)
    "FRD": ("Fredericksburg, US", 38.20, -77.37),
    "BOU": ("Boulder, US", 40.14, -105.24),
    "NEW": ("Newport, US", 48.27, -117.12),
    "SIT": ("Sitka, US", 57.06, -135.33),
    "BRW": ("Barrow, US", 71.32, -156.62),
}
INTERMAGNET = {
    "TIH": ("Tihany, HU", 46.90, 17.89),
    "NGK": ("Niemegk, DE", 52.07, 12.68),
    "ESK": ("Eskdalemuir, UK", 55.31, -3.21),
    "HAD": ("Hartland, UK", 50.98, -4.48),
    "LER": ("Lerwick, UK", 60.14, -1.18),
    "ABK": ("Abisko, SE", 68.36, 18.82),
    "OTT": ("Ottawa, CA", 45.40, -75.55),
}


def http_get(url, timeout=40):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def good(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (f != f or abs(f) >= 88888) else f


# ---------- USGS ----------
def fetch_usgs(iaga, start, end):
    base = "https://geomag.usgs.gov/ws/data/"
    for elements in ("X,Y", "H,E"):
        q = urllib.parse.urlencode({
            "id": iaga, "starttime": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endtime": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "elements": elements,
            "sampling_period": 60, "type": "variation", "format": "json",
        })
        d = json.loads(http_get(f"{base}?{q}"))
        times = d.get("times") or []
        vals = {v.get("id") or v.get("metadata", {}).get("element"): v.get("values") for v in d.get("values", [])}
        a_key, b_key = elements.split(",")
        a, b = vals.get(a_key), vals.get(b_key)
        if times and a and b:
            pts = []
            for t, x, y in zip(times, a, b):
                x, y = good(x), good(y)
                if x is not None and y is not None:
                    ts = datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                    pts.append((int(ts.timestamp()), x, y))
            if pts:
                return pts
    raise RuntimeError("no X/Y or H/E values")


# ---------- INTERMAGNET (BGS GIN, IAGA-2002 text) ----------
def fetch_gin(iaga, start, end):
    days = (end.date() - start.date()).days + 1
    q = urllib.parse.urlencode({
        "Request": "GetData", "format": "iaga2002", "testObsys": 0,
        "observatoryIagaCode": iaga, "samplesPerDay": "minute",
        "publicationState": "best-avail", "orientation": "native",
        "dataStartDate": start.strftime("%Y-%m-%d"), "dataDuration": days,
    })
    text = http_get(f"https://imag-data.bgs.ac.uk/GIN_V1/GINServices?{q}")
    cols, pts = None, []
    for line in text.splitlines():
        if line.startswith("DATE"):
            cols = [c.replace(iaga, "", 1) for c in line.split()[3:7]]  # e.g. X Y Z F or H D Z F
            continue
        if cols is None or not line[:4].isdigit():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        ts = datetime.strptime(parts[0] + " " + parts[1][:8], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        v = dict(zip(cols, (good(p) for p in parts[3:7])))
        if v.get("X") is not None and v.get("Y") is not None:
            x, y = v["X"], v["Y"]
        elif v.get("H") is not None and v.get("D") is not None:
            d = math.radians(v["D"] / 60.0)  # D is in minutes of arc
            x, y = v["H"] * math.cos(d), v["H"] * math.sin(d)
        elif v.get("H") is not None and v.get("E") is not None:
            x, y = v["H"], v["E"]
        else:
            continue
        pts.append((int(ts.timestamp()), x, y))
    if cols is None:
        raise RuntimeError(f"no IAGA-2002 data: {text[:150]!r}")
    return pts


def dbdt(pts):
    pts = sorted(pts)
    out = []
    for (t0, x0, y0), (t1, x1, y1) in zip(pts, pts[1:]):
        dt_min = (t1 - t0) / 60.0
        if 0 < dt_min <= 2:
            out.append((t1, math.hypot(x1 - x0, y1 - y0) / dt_min))
    return out


def iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main():
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = now - timedelta(hours=WINDOW_HOURS)
    res = {"fetched_at": iso(int(now.timestamp())), "data_time": None, "stations": [], "errors": []}
    newest = 0
    jobs = [(k, v, "USGS", fetch_usgs) for k, v in USGS.items()] + \
           [(k, v, "INTERMAGNET", fetch_gin) for k, v in INTERMAGNET.items()]
    for iaga, (name, lat, lng), src, fn in jobs:
        try:
            pts = [p for p in fn(iaga, start, now) if p[0] >= start.timestamp()]
            series = dbdt(pts)
            if not series:
                raise RuntimeError("no samples in window")
            last_t = series[-1][0]
            lag = round((now.timestamp() - last_t) / 60)
            if lag > MAX_LAG_MIN:
                raise RuntimeError(f"newest sample {lag} min old")
            recent = [v for t, v in series if t > last_t - PEAK_MINUTES * 60]
            newest = max(newest, last_t)
            res["stations"].append({
                "id": iaga, "name": name, "source": src, "lat": lat, "lng": lng,
                "dbdt": round(max(recent), 1), "dbdt_now": round(series[-1][1], 1),
                "time": iso(last_t), "lag_min": lag,
            })
            print(f"{iaga} ({src}): {len(series)} pts, peak {max(recent):.1f} nT/min, lag {lag} min")
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"{iaga}: {e}")
            print(f"{iaga} ({src}): failed: {str(e)[:200]}")

    if newest:
        res["data_time"] = iso(newest)
    else:
        try:
            with open(OUT) as f:
                prev = json.load(f)
            if prev.get("stations"):
                res.update(stations=prev["stations"], data_time=prev.get("data_time"), stale=True)
                print("no fresh data, keeping previous stations (stale)")
        except Exception:  # noqa: BLE001
            pass

    with open(OUT + ".tmp", "w") as f:
        json.dump(res, f, separators=(",", ":"))
    os.replace(OUT + ".tmp", OUT)
    print(f"{len(res['stations'])} stations written, data_time {res['data_time']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
