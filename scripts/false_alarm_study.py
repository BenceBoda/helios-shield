#!/usr/bin/env python3
"""How often would the V1 rule have warned, missed, or cried wolf over a full year?

Inputs (no SuperMAG needed):
  - NASA OMNI 1-minute solar wind (CDAWeb HAPI)            -> rule alerts
  - USGS + INTERMAGNET 1-minute ground magnetometer data   -> what actually happened (dB/dt)
Output: analysis/false_alarms_<year>.json and a printed summary.

Definitions
  Alert:   Ey = V*Bs >= EY mV/m for 10 consecutive minutes (OMNI, bow-shock time),
           moved earlier by the L1 transit time (1.41e6 km / V). Alerts closer than
           6 h to the previous one belong to the same alert episode.
  Event:   an hour in which any mid-latitude station's dB/dt reached DBDT nT/min.
           Hours closer than 6 h belong to the same event.
  Warned:  an alert episode started 0-12 h before the event's first hour.
  Late:    the first alert came during the first 3 h of the event.
  Missed:  no alert from 12 h before to 3 h after the event start.
  False alarm: an alert episode with no event from its start to 12 h later.
  An hour only counts if a second station saw at least a third of the threshold at the same
  time; single-station jumps and values above MAX_PLAUSIBLE nT/min are instrument artefacts.
"""
import csv
import io
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_live_magnetometers import dbdt, fetch_gin, fetch_usgs  # noqa: E402

YEAR = int(os.environ.get("STUDY_YEAR", "2024"))
HAPI = "https://cdaweb.gsfc.nasa.gov/hapi"
L1_KM = 1.41e6
SUSTAIN = 10
EPISODE_GAP_H = 6
WARN_BEFORE_H = 12
LATE_AFTER_H = 3
EY_LEVELS = [5.0, 10.0, 15.0]
DBDT_LEVELS = [100.0, 300.0]
MAX_PLAUSIBLE = 5000.0  # nT/min; far above anything recorded at mid-latitudes
STATIONS = [("FRD", "usgs"), ("BOU", "usgs"), ("NEW", "usgs"), ("SIT", "usgs"),
            ("NGK", "gin"), ("ESK", "gin"), ("HAD", "gin"), ("LER", "gin")]


def http(url, timeout=120):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "helios-shield-demo"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def omni_year(year):
    import urllib.parse
    rows = []
    for m in range(1, 13):
        a = datetime(year, m, 1, tzinfo=timezone.utc)
        b = datetime(year + (m == 12), m % 12 + 1, 1, tzinfo=timezone.utc)
        q = urllib.parse.urlencode({"id": "OMNI_HRO2_1MIN", "parameters": "BZ_GSM,flow_speed", "format": "csv",
                                    "time.min": a.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    "time.max": b.strftime("%Y-%m-%dT%H:%M:%SZ")})
        for attempt in range(3):
            try:
                text = http(f"{HAPI}/data?{q}")
                break
            except Exception as e:  # noqa: BLE001
                print(f"OMNI {a:%Y-%m} attempt {attempt + 1}: {e}")
                time.sleep(10)
        else:
            continue
        n = 0
        for r in csv.reader(io.StringIO(text)):
            try:
                t = datetime.strptime(r[0][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                bz, v = float(r[1]), float(r[2])
            except (ValueError, IndexError):
                continue
            rows.append((int(t.timestamp()), bz if abs(bz) < 999 else None, v if 0 < v < 9999 else None))
            n += 1
        print(f"OMNI {a:%Y-%m}: {n} rows")
    return rows


def alerts(rows, ey_thr):
    """Return alert episode start times (L1 time, epoch s)."""
    out, run, last = [], 0, None
    for t, bz, v in rows:
        hit = bz is not None and v is not None and v * max(0.0, -bz) * 1e-3 >= ey_thr
        run = run + 1 if hit else 0
        if run >= SUSTAIN:
            t_l1 = t - int(L1_KM / v)
            if last is None or t_l1 - last > EPISODE_GAP_H * 3600:
                out.append(t_l1)
            last = t_l1
    return out


def ground_hourly(year):
    """Per-station hourly max dB/dt: {hour_epoch: {station: value}} and coverage per station.
    Cached in analysis/ground_hourly_<year>.json so reruns skip the download."""
    cache = f"analysis/ground_hourly_{year}.json"
    if os.path.exists(cache):
        with open(cache) as f:
            c = json.load(f)
        print(f"using cached ground data ({cache})")
        return {int(h): v for h, v in c["hourly"].items()}, c["coverage"]
    hourly, cover = {}, {}
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    for iaga, src in STATIONS:
        got = 0
        t = start
        while t < end:
            u = min(t + timedelta(days=10), end)
            pts = None
            for attempt in range(3):
                try:
                    if src == "usgs":
                        try:  # definitive data has the instrument glitches cleaned out
                            pts = fetch_usgs(iaga, t, u, "definitive")
                        except Exception:  # noqa: BLE001
                            pts = fetch_usgs(iaga, t, u, "variation")
                    else:
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
                    row = hourly.setdefault(h, {})
                    if v > row.get(iaga, 0):
                        row[iaga] = round(v, 1)
                    got += 1
            t = u
        cover[iaga] = round(got / ((end - start).total_seconds() / 60), 3)
        print(f"{iaga}: coverage {cover[iaga]:.0%}")
    with open(cache, "w") as f:
        json.dump({"hourly": hourly, "coverage": cover}, f, separators=(",", ":"))
    return hourly, cover


def confirmed(row, thr):
    """(value, station) if the hour is a real event: the top station reaches thr AND a second
    station sees at least thr/3 at the same time. Single-station jumps are instrument artefacts."""
    vals = sorted(((v, s) for s, v in row.items() if v <= MAX_PLAUSIBLE), reverse=True)
    if len(vals) >= 2 and vals[0][0] >= thr and vals[1][0] >= thr / 3:
        return vals[0]
    return None


def events(hourly, thr):
    hits = {h: confirmed(row, thr) for h, row in hourly.items()}
    hours = sorted(h for h, c in hits.items() if c)
    ev = []
    for h in hours:
        v, st = hits[h]
        if ev and h - ev[-1]["end"] <= EPISODE_GAP_H * 3600:
            ev[-1]["end"] = h
            if v > ev[-1]["peak"]:
                ev[-1]["peak"], ev[-1]["station"] = v, st
        else:
            ev.append({"start": h, "end": h, "peak": v, "station": st})
    return ev


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def evaluate(al, ev):
    res = {"warned": [], "late": [], "missed": [], "false_alarms": []}
    for e in ev:
        before = [a for a in al if e["start"] - WARN_BEFORE_H * 3600 <= a < e["start"]]
        during = [a for a in al if e["start"] <= a < e["start"] + LATE_AFTER_H * 3600]
        item = {"event_start": iso(e["start"]), "peak_dbdt": round(e["peak"]), "station": e["station"]}
        if before:
            item["lead_min"] = round((e["start"] - before[0]) / 60)
            res["warned"].append(item)
        elif during:
            res["late"].append(item)
        else:
            res["missed"].append(item)
    for a in al:
        hit = any(a <= e["start"] <= a + WARN_BEFORE_H * 3600 or e["start"] <= a <= e["end"] + 3600 for e in ev)
        if not hit:
            res["false_alarms"].append(iso(a))
    return res


def main():
    os.makedirs("analysis", exist_ok=True)
    rows = omni_year(YEAR)
    hourly, cover = ground_hourly(YEAR)
    out = {"year": YEAR, "station_coverage": cover, "omni_rows": len(rows),
           "definitions": __doc__.split("Definitions")[1].strip(), "results": []}
    print("\nEY mV/m | dB/dt nT/min | alerts | events | warned | late | missed | false alarms")
    for ey in EY_LEVELS:
        al = alerts(rows, ey)
        for thr in DBDT_LEVELS:
            ev = events(hourly, thr)
            r = evaluate(al, ev)
            out["results"].append({"ey": ey, "dbdt": thr, "alerts": len(al), "events": len(ev),
                                   **{k: len(v) for k, v in r.items()}, "detail": r})
            print(f"{ey:7.0f} | {thr:12.0f} | {len(al):6d} | {len(ev):6d} | {len(r['warned']):6d} | "
                  f"{len(r['late']):4d} | {len(r['missed']):6d} | {len(r['false_alarms']):12d}")
    with open(f"analysis/false_alarms_{YEAR}.json", "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
