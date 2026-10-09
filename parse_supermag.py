#!/usr/bin/env python3
import json, sys, math, datetime

ts = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

try:
    with open('supermag_raw.json') as f:
        raw = f.read().strip()
    if not raw:
        raise ValueError('empty file')
    data = json.loads(raw)
except Exception as e:
    print(json.dumps({'stations': [], 'fetched_at': ts, 'error': str(e)}))
    sys.exit(0)

if not isinstance(data, list) or len(data) == 0:
    print(json.dumps({
        'stations': [],
        'fetched_at': ts,
        'raw_type': type(data).__name__,
        'raw_preview': str(data)[:300]
    }))
    sys.exit(0)

out = {}
for r in data:
    sid = (r.get('iaga') or r.get('IAGA') or r.get('station') or '').upper().strip()
    if not sid:
        continue
    try:
        n = float(r.get('dbn_nez') or r.get('Bn') or r.get('bn') or 0)
    except (TypeError, ValueError):
        n = 0.0
    try:
        e = float(r.get('dbe_nez') or r.get('Be') or r.get('be') or 0)
    except (TypeError, ValueError):
        e = 0.0
    dbdt = round(math.sqrt(n * n + e * e) * 10) / 10
    if sid not in out or dbdt > out[sid]['dbdt']:
        try:
            lat = float(r.get('geolatitude') or r.get('lat') or 0)
        except (TypeError, ValueError):
            lat = 0.0
        try:
            lng = float(r.get('geolongitude') or r.get('lng') or r.get('lon') or 0)
        except (TypeError, ValueError):
            lng = 0.0
        out[sid] = {
            'id': sid,
            'lat': lat,
            'lng': lng,
            'dbdt': dbdt,
            'time': str(r.get('tval') or r.get('time') or '')
        }

print(json.dumps({'stations': list(out.values()), 'fetched_at': ts}))
