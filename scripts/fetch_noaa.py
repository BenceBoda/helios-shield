#!/usr/bin/env python3
"""Fetch NOAA SWPC data: planetary Kp (kp.json) and L1 solar wind (bz.json).

bz.json format:
  {"fetched_at": ISO, "source": url, "data": [{"t": "YYYY-MM-DDTHH:MM", "bz": nT, "bt": nT,
                                              "speed": km/s, "density": p/cc}, ...]}
On any failure the previous file is left untouched, so the demo never shows a broken file.
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone

BASE = "https://services.swpc.noaa.gov"
KP_URL = BASE + "/json/planetary_k_index_1m.json"
MAG_URLS = [BASE + "/json/rtsw/rtsw_mag_1m.json"]
PLASMA_URLS = [BASE + "/json/rtsw/rtsw_wind_1m.json"]
KEEP_MINUTES = 6 * 60


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "helios-shield-demo"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def as_rows(data):
    """NOAA 'products' are arrays of arrays with a header row; 'json' feeds are arrays of objects."""
    if isinstance(data, list) and data and isinstance(data[0], list):
        head = data[0]
        return [dict(zip(head, r)) for r in data[1:]]
    return data if isinstance(data, list) else []


def minute_key(t):
    return str(t).replace(" ", "T")[:16] if t else None


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if abs(f) >= 9999 else f


def first_ok(urls):
    for u in urls:
        try:
            rows = as_rows(get_json(u))
            if rows:
                return u, rows
            print(f"empty response: {u}")
        except Exception as e:  # noqa: BLE001
            print(f"failed: {u}: {e}")
    return None, []


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Kp
    try:
        kp = get_json(KP_URL)
        if isinstance(kp, list) and kp:
            write_json("kp.json", kp)
            print(f"kp.json: {len(kp)} rows")
        else:
            print("kp: empty response, keeping previous file")
    except Exception as e:  # noqa: BLE001
        print(f"kp failed, keeping previous file: {e}")

    # Solar wind
    mag_src, mag = first_ok(MAG_URLS)
    _, plasma = first_ok(PLASMA_URLS)
    if not mag:
        print("solar wind mag unavailable, keeping previous bz.json")
        return 0

    merged = {}
    for r in mag:
        k = minute_key(r.get("time_tag"))
        bz = num(r.get("bz_gsm"))
        if not k or bz is None:
            continue
        if r.get("active") is False:  # rtsw feed lists several spacecraft; keep the active one
            continue
        merged[k] = {"t": k, "bz": round(bz, 2), "bt": num(r.get("bt")), "speed": None, "density": None}
    for r in plasma:
        k = minute_key(r.get("time_tag"))
        if k in merged and r.get("active") is not False:
            merged[k]["speed"] = num(r.get("speed", r.get("proton_speed")))
            merged[k]["density"] = num(r.get("density", r.get("proton_density")))

    data = [merged[k] for k in sorted(merged)][-KEEP_MINUTES:]
    if not data:
        print("no usable solar wind rows, keeping previous bz.json")
        return 0
    write_json("bz.json", {"fetched_at": now, "source": mag_src, "data": data})
    last = data[-1]
    print(f"bz.json: {len(data)} rows, latest {last['t']} Bz={last['bz']} speed={last['speed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
