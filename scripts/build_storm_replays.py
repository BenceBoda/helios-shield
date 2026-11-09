#!/usr/bin/env python3
"""Build measured storm replay data -> storms/<key>.json

Sources:
  - SuperMAG 1-minute magnetometer data (needs env SM_LOGON)  -> station dB/dt
  - NASA CDAWeb HAPI, OMNI solar wind (1-min, or hourly before 1995) -> Bz, speed
  - GFZ Potsdam Kp index (3-hourly)

Alert rule (V1, rule based):
  WARNING when the solar-wind electric field Ey = V * Bs (Bs = southward Bz) stays
  >= EY_WARN mV/m for SUSTAIN_MIN minutes. OMNI is time-shifted to Earth's bow shock,
  so the time the same signal was seen at L1 is earlier by ~1.41e6 km / V.
"Danger" = the first minute any station's measured dB/dt reaches DANGER_DBDT nT/min.
"""
import csv
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_supermag import dbdt_series, fetch_station  # noqa: E402

EY_WARN = 10.0          # mV/m
SUSTAIN_MIN = 10
BZ_WARN = 10.0          # nT, fallback rule when solar wind speed is missing
DANGER_DBDT = 300.0     # nT/min
DANGER_MAX_LAT = 62.0   # only mid/sub-auroral stations count: polar sites spike even in moderate storms
L1_TO_BOWSHOCK_KM = 1.41e6
BINS = 24
UA = {"User-Agent": "helios-shield-demo"}
HAPI = "https://cdaweb.gsfc.nasa.gov/hapi"

STORMS = {
    "quebec1989": {"start": "1989-03-13T00:00", "hours": 24},
    "halloween2003": {"start": "2003-10-29T00:00", "hours": 48},
    "gannon2024": {"start": "2024-05-10T12:00", "hours": 24},
}

STATIONS = {  # IAGA: (name, lat, lon)
    "VDQ": ("Val d'Or, QC", 48.0, -77.8), "OTT": ("Ottawa, ON", 45.4, -75.6),
    "YKC": ("Yellowknife, NT", 62.5, -114.5), "BRW": ("Barrow, AK", 71.3, -156.6),
    "MEA": ("Meanook, AB", 54.6, -113.3), "STJ": ("St. John's, NL", 47.6, -52.7),
    "FRD": ("Fredericksburg, VA", 38.2, -77.4), "HAD": ("Hartland, UK", 51.0, -4.5),
    "SIT": ("Sitka, AK", 57.1, -135.3), "ABK": ("Abisko, SE", 68.4, 18.8),
    "TRO": ("Tromsø, NO", 69.7, 18.9), "KIR": ("Kiruna, SE", 67.8, 20.4),
    "SOD": ("Sodankylä, FI", 67.4, 26.6), "UPS": ("Uppsala, SE", 59.9, 17.4),
    "BOR": ("Borok, RU", 58.0, 38.3), "HER": ("Hermanus, ZA", -34.4, 19.2),
    "GNG": ("Gingin, AU", -31.4, 115.7), "ESK": ("Eskdalemuir, UK", 55.3, -3.2),
    "BFE": ("Brorfelde, DK", 55.6, 11.7), "NEW": ("Newport, WA", 48.3, -117.1),
    "CNB": ("Canberra, AU", -35.3, 149.4), "HON": ("Honolulu, HI", 21.3, -158.0),
    "THY": ("Tihany, HU", 46.9, 17.9), "NGK": ("Niemegk, DE", 52.1, 12.7),
    "LER": ("Lerwick, UK", 60.1, -1.2),
}


def get(url, timeout=90):
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"HTTP {e.code} for {url[:160]}: {body}") from None


def ts(dt):
    return int(dt.timestamp())


def isoz(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


# ---------- Solar wind (OMNI via HAPI) ----------
def hapi_params(ds):
    info = json.loads(get(f"{HAPI}/info?id={ds}"))
    return [p["name"] for p in info.get("parameters", [])]


def pick(names, patterns):
    for pat in patterns:
        for n in names:
            if re.fullmatch(pat, n, re.I):
                return n
    return None


def fetch_omni(start, end):
    """Returns (dataset, [(t, bz, v)]). Tries 1-min first, then hourly."""
    for ds, cadence in (("OMNI_HRO2_1MIN", 60), ("OMNI2_H0_MRG1HR", 3600)):
        try:
            names = hapi_params(ds)
            bz = pick(names, [r"BZ_GSM", r"BZ_GSM\d*", r"BZ.*GSM.*"])
            v = pick(names, [r"flow_speed", r"V\d*", r"V_?SW.*", r"Plasma_?speed.*"])
            if not bz or not v:
                print(f"{ds}: could not find Bz/V among {names}")
                continue
            t0, t1 = start.strftime("%Y-%m-%dT%H:%M:%SZ"), end.strftime("%Y-%m-%dT%H:%M:%SZ")
            text = None
            for style in ({"start": t0, "stop": t1}, {"time.min": t0, "time.max": t1}):
                q = urllib.parse.urlencode({"id": ds, "parameters": f"{bz},{v}", "format": "csv", **style})
                try:
                    text = get(f"{HAPI}/data?{q}")
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"{ds} {list(style)[0]}: {e}")
            if text is None:
                continue
            rows = []
            for r in csv.reader(io.StringIO(text)):
                if len(r) < 3:
                    continue
                try:
                    t = datetime.strptime(r[0][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                    b, s = float(r[1]), float(r[2])
                except ValueError:
                    continue
                if abs(b) < 999:  # keep Bz even when speed is missing (e.g. ACE plasma saturated in 2003)
                    rows.append((ts(t), b, s if 0 < s < 9999 else None))
            print(f"{ds}: {len(rows)} valid rows ({bz}, {v})")
            if rows:
                return ds, cadence, rows
        except Exception as e:  # noqa: BLE001
            print(f"{ds}: failed: {e}")
    return None, None, []


# ---------- Kp (GFZ) ----------
def fetch_kp(start, end):
    q = urllib.parse.urlencode({
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"), "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"), "index": "Kp"})
    d = json.loads(get(f"https://kp.gfz-potsdam.de/app/json/?{q}"))
    out = []
    for t, k in zip(d.get("datetime", []), d.get("Kp", [])):
        tt = datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
        out.append((ts(tt), float(k)))
    return out


# ---------- analysis ----------
def warning_time(sw, cadence):
    """First time the rule holds for SUSTAIN_MIN (bow-shock time). Returns (t, v, ey, rule).
    Rule: Ey = V*Bs >= EY_WARN mV/m; where speed is missing, Bz <= -BZ_WARN nT instead."""
    need = max(1, SUSTAIN_MIN * 60 // cadence)
    run = 0
    for t, bz, v in sw:
        if v is not None:
            hit, ey, rule = v * max(0.0, -bz) * 1e-3 >= EY_WARN, v * max(0.0, -bz) * 1e-3, "Ey"
        else:
            hit, ey, rule = bz <= -BZ_WARN, None, "Bz"
        run = run + 1 if hit else 0
        if run >= need:
            return t, v, ey, rule
    return None, None, None, None


def main():
    logon = os.environ.get("SM_LOGON", "").strip()
    os.makedirs("storms", exist_ok=True)
    for key, cfg in STORMS.items():
        start = datetime.strptime(cfg["start"], "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
        end = start + timedelta(hours=cfg["hours"])
        bin_s = cfg["hours"] * 3600 // BINS
        edges = [ts(start) + i * bin_s for i in range(BINS)]
        print(f"\n=== {key}: {start} .. {end}")

        def bin_of(t):
            i = (t - ts(start)) // bin_s
            return int(i) if 0 <= i < BINS else None

        out = {"key": key, "start": isoz(ts(start)), "hours": cfg["hours"], "bin_minutes": bin_s // 60,
               "labels": [datetime.fromtimestamp(e, timezone.utc).strftime("%d %H:%M") for e in edges],
               "sources": {}}

        # Kp
        try:
            kp = fetch_kp(start - timedelta(hours=3), end)
            arr = []
            for e in edges:
                vals = [k for t, k in kp if t <= e]
                arr.append(vals[-1] if vals else None)
            out["kp"] = arr
            out["sources"]["kp"] = "GFZ Potsdam"
            print(f"Kp: max {max(k for k in arr if k is not None)}")
        except Exception as e:  # noqa: BLE001
            print(f"Kp failed: {e}")

        # Solar wind
        ds, cadence, sw = fetch_omni(start, end)
        if sw:
            sums = [[0.0, 0] for _ in range(BINS)]
            for t, b, _ in sw:
                i = bin_of(t)
                if i is not None:
                    sums[i][0] += b
                    sums[i][1] += 1
            out["bz"] = [round(s / n, 1) if n else None for s, n in sums]
            out["bz_min"] = round(min(b for _, b, _ in sw), 1)
            vs = [v for _, _, v in sw if v is not None]
            out["v_max"] = round(max(vs)) if vs else None
            out["sources"]["solar_wind"] = f"NASA OMNI ({ds})"
            wt, wv, wey, rule = warning_time(sw, cadence)
            if wt:
                # Without a measured speed we cannot estimate the L1 lead: count from bow-shock time (conservative)
                lead = round(L1_TO_BOWSHOCK_KM / wv / 60) if wv else 0
                out["warning"] = {"bowshock_time": isoz(wt), "l1_time": isoz(wt - lead * 60),
                                  "l1_lead_min": lead, "speed": round(wv) if wv else None,
                                  "ey": round(wey, 1) if wey is not None else None, "rule": rule,
                                  "cadence_s": cadence}
                print(f"warning: bow shock {isoz(wt)}, L1 {isoz(wt - lead * 60)} (lead {lead} min, V={wv}, rule {rule})")

        # SuperMAG stations
        stations, dmax, first_danger = [], [0.0] * BINS, None
        for iaga, (name, lat, lng) in STATIONS.items() if logon else []:
            series, err = None, None
            for attempt in range(3):
                try:
                    series = dbdt_series(fetch_station(logon, iaga, start.strftime("%Y-%m-%dT%H:%M"),
                                                       cfg["hours"] * 3600))
                    break
                except Exception as e:  # noqa: BLE001
                    err = e
                    time.sleep(5 * (attempt + 1))
            if series is None:
                print(f"{iaga}: failed: {str(err)[:120]}")
                continue
            if not series:
                continue
            pt, pv = max(series, key=lambda x: x[1])
            stations.append({"id": iaga, "name": name, "lat": lat, "lng": lng,
                             "dbdt": round(pv), "peak_time": isoz(pt)})
            for t, v in series:
                i = bin_of(t)
                if i is not None:
                    dmax[i] = max(dmax[i], v)
                if abs(lat) <= DANGER_MAX_LAT and v >= DANGER_DBDT and (first_danger is None or t < first_danger[0]):
                    first_danger = (t, iaga, v)
            print(f"{iaga}: peak {pv:.0f} nT/min at {isoz(pt)} (spikes removed: {getattr(dbdt_series, 'last_removed', 0)})")
        if stations:
            stations.sort(key=lambda s: -s["dbdt"])
            out["stations"] = stations
            out["dbdt_max"] = [round(v) for v in dmax]
            out["sources"]["magnetometers"] = "SuperMAG (Gjerloev 2012)"
        if first_danger:
            out["danger"] = {"time": isoz(first_danger[0]), "station": first_danger[1],
                             "dbdt": round(first_danger[2]), "threshold": DANGER_DBDT}
            if out.get("warning"):
                l1 = datetime.strptime(out["warning"]["l1_time"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
                out["warning"]["minutes_before_danger"] = round((first_danger[0] - ts(l1)) / 60)
            print(f"danger: {out['danger']}  -> warning lead {out.get('warning', {}).get('minutes_before_danger')}")

        path = f"storms/{key}.json"
        try:
            with open(path) as f:
                prev = json.load(f)
        except Exception:  # noqa: BLE001
            prev = {}
        # Never replace good data with a degraded run (e.g. SuperMAG outage)
        if len(out.get("stations", [])) < len(prev.get("stations", [])):
            print(f"{key}: only {len(out.get('stations', []))} stations vs {len(prev['stations'])} before - keeping previous file")
            continue
        with open(path, "w") as f:
            json.dump(out, f, separators=(",", ":"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
