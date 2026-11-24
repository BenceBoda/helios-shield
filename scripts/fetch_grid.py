#!/usr/bin/env python3
"""Hungarian high-voltage transmission lines (220, 400, 750 kV) from OpenStreetMap via Overpass.

Output: sandbox/hu_grid.geojson, one LineString per OSM way, with voltage (kV), name,
operator and length (km). OSM is crowd-mapped: good for line routes, says nothing about
transformers, earthing or line resistance, so it cannot give GIC on its own.
Data (c) OpenStreetMap contributors, ODbL.
"""
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

QUERY = """[out:json][timeout:180];
area["ISO3166-1"="HU"][admin_level=2]->.hu;
way["power"="line"]["voltage"~"(^|;)(220000|400000|750000)(;|$)"](area.hu);
out tags geom;"""
SERVERS = ["https://overpass-api.de/api/interpreter",
           "https://overpass.kumi.systems/api/interpreter",
           "https://overpass.private.coffee/api/interpreter"]
OUT = "sandbox/hu_grid.geojson"


def km(a, b):
    """Great-circle distance between (lat, lon) points in km."""
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dl = math.radians(b[1] - a[1])
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def main():
    data = None
    for url in SERVERS:
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": QUERY}).encode(),
                                         headers={"User-Agent": "helios-shield-demo/0.1 (GitHub Actions)"})
            with urllib.request.urlopen(req, timeout=240) as r:
                data = json.load(r)
            print(f"{url}: {len(data.get('elements', []))} ways")
            break
        except Exception as e:  # noqa: BLE001
            print(f"{url}: {str(e)[:150]}")
            time.sleep(5)
    if not data:
        return 1
    feats = []
    for el in data["elements"]:
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        tags = el.get("tags", {})
        volts = [int(v) for v in tags.get("voltage", "").split(";") if v.strip().isdigit()]
        kv = max(volts) // 1000 if volts else None
        coords = [[round(p["lon"], 5), round(p["lat"], 5)] for p in geom]
        length = sum(km((a[1], a[0]), (b[1], b[0])) for a, b in zip(coords, coords[1:]))
        feats.append({"type": "Feature",
                      "properties": {"id": el["id"], "kv": kv, "name": tags.get("name"),
                                     "operator": tags.get("operator"), "circuits": tags.get("circuits"),
                                     "km": round(length, 2)},
                      "geometry": {"type": "LineString", "coordinates": coords}})
    os.makedirs("sandbox", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"type": "FeatureCollection",
                   "attribution": "(c) OpenStreetMap contributors, ODbL",
                   "fetched": time.strftime("%Y-%m-%d"), "features": feats}, f, separators=(",", ":"))
    by_kv = {}
    for ft in feats:
        k = ft["properties"]["kv"]
        by_kv[k] = by_kv.get(k, 0) + ft["properties"]["km"]
    print(f"{OUT}: {len(feats)} ways; km by kV: { {k: round(v) for k, v in sorted(by_kv.items(), key=str)} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
