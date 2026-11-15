"""Training-only baseline runner. No evaluation-file access or tuning on test."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from .baselines import evaluate, nearest_neighbor, two_opt
from .prepare import read_city_map


def objects(path):
    """Stream members of a top-level JSON object, keeping one route in memory."""
    decoder = json.JSONDecoder()
    with path.open() as handle:
        buffer = ''
        eof = False
        def fill():
            nonlocal buffer, eof
            chunk = handle.read(1024 * 1024)
            buffer += chunk
            eof = not chunk
        def token():
            nonlocal buffer
            while True:
                buffer = buffer.lstrip()
                if buffer or eof:
                    return buffer[:1]
                fill()
        def consume(expected):
            nonlocal buffer
            if token() != expected:
                raise ValueError(f'{path}: expected {expected}')
            buffer = buffer[1:]
        def value():
            nonlocal buffer
            token()
            while True:
                try:
                    result, end = decoder.raw_decode(buffer)
                    buffer = buffer[end:]
                    return result
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
        consume('{')
        if token() != '}':
            while True:
                key = value()
                if not isinstance(key, str):
                    raise ValueError('JSON object key must be a string')
                consume(':')
                yield key, value()
                if token() == '}':
                    break
                consume(',')
        consume('}')
        if token():
            raise ValueError('trailing JSON data')


def run(raw_dir, city_map, output, limit=0, passes=5):
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    source = raw_dir / 'almrrc2021-data-training' / 'model_build_inputs'
    routes = dict(objects(source / 'route_data.json'))
    sequences = dict(objects(source / 'actual_sequences.json'))
    if set(routes) != set(sequences):
        raise ValueError('route/sequence IDs differ')
    cities = read_city_map(city_map)
    if any(r['station_code'] not in cities for r in routes.values()):
        raise ValueError('unmapped station')
    selected = set(sorted(routes)[:limit] if limit else routes)
    output.mkdir(parents=True)
    manifest = {}
    for path in [source / n for n in ('route_data.json', 'actual_sequences.json',
                                      'package_data.json', 'travel_times.json')] + [city_map]:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        manifest[str(path)] = {'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}
    config = dict(split='training', limit=limit, max_2opt_passes=passes,
                  return_to_depot=True, windows_at='service_start',
                  mapping_status='provisional', selection='sorted route IDs',
                  deterministic=True, official_amazon_score=False, manifest=manifest)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    folds = {}
    for city in sorted(set(cities.values())):
        folds[city] = {'train': sorted(r for r in routes if cities[routes[r]['station_code']] != city),
                       'validation': sorted(r for r in routes if cities[routes[r]['station_code']] == city)}
    (output / 'training_folds.json').write_text(json.dumps(folds, indent=2))
    totals = defaultdict(lambda: defaultdict(float))
    seen = set()
    with tempfile.TemporaryDirectory(prefix='lastmile-') as temporary:
        database = sqlite3.connect(str(Path(temporary) / 'packages.sqlite'))
        database.execute('CREATE TABLE packages (id TEXT PRIMARY KEY, payload TEXT)')
        package_ids = set()
        for rid, payload in objects(source / 'package_data.json'):
            if rid in package_ids:
                raise ValueError('duplicate package route ID')
            package_ids.add(rid)
            if rid in selected:
                database.execute('INSERT INTO packages VALUES (?, ?)', (rid, json.dumps(payload)))
        database.commit()
        if package_ids != set(routes):
            raise ValueError('package/route IDs differ')
        with (output / 'results.csv').open('w', newline='') as results, (output / 'sequences.jsonl').open('w') as solutions:
            writer = None
            matrix_ids = set()
            for rid, costs in objects(source / 'travel_times.json'):
                if rid in matrix_ids:
                    raise ValueError('duplicate matrix route ID')
                matrix_ids.add(rid)
                if rid not in selected:
                    continue
                route = routes[rid]
                packages = json.loads(database.execute('SELECT payload FROM packages WHERE id=?', (rid,)).fetchone()[0])
                ranks = sequences[rid]['actual']
                if sorted(ranks.values()) != list(range(len(route['stops']))):
                    raise ValueError('invalid observed ranks')
                observed = sorted(ranks, key=ranks.get)
                reference = evaluate(observed, route, packages, costs)
                depot = observed[0]
                for method in ('observed', 'nearest_neighbor', 'nearest_neighbor_2opt'):
                    begin = time.perf_counter()
                    if method == 'observed':
                        sequence = observed
                    else:
                        sequence = nearest_neighbor(costs, depot)
                        if method.endswith('2opt'):
                            sequence = two_opt(sequence, costs, passes)
                    runtime = time.perf_counter() - begin
                    metrics = evaluate(sequence, route, packages, costs)
                    row = dict(route_id=rid, city=cities[route['station_code']],
                               route_quality=route['route_score'], method=method,
                               stops=len(sequence), runtime_seconds=runtime, **metrics)
                    row['travel_delta_vs_observed_seconds'] = metrics['travel_seconds'] - reference['travel_seconds']
                    if writer is None:
                        writer = csv.DictWriter(results, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    solutions.write(json.dumps(dict(route_id=rid, method=method, sequence=sequence)) + '\n')
                    for group in (method, method + '/' + row['city']):
                        totals[group]['routes'] += 1
                        for key in (*metrics, 'runtime_seconds', 'travel_delta_vs_observed_seconds'):
                            totals[group][key] += row[key]
                seen.add(rid)
                if len(seen) % 100 == 0:
                    print(f'Completed {len(seen)}/{len(selected)} training routes', flush=True)
            if matrix_ids != set(routes) or seen != selected:
                raise ValueError('matrix/selected route IDs differ')
        database.close()
    summary = {group: {'routes': int(values['routes']), **{
        'mean_' + key: value / values['routes'] for key, value in values.items() if key != 'routes'
    }} for group, values in sorted(totals.items())}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'status': 'complete', 'training_routes': len(seen), 'output': str(output)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, default=Path('data/raw/amazon'))
    parser.add_argument('--city-map', type=Path, default=Path('data/metadata/station_city_map.csv'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--passes', type=int, default=5)
    args = parser.parse_args()
    if args.limit < 0 or args.passes < 0:
        parser.error('limit and passes must be nonnegative')
    run(args.raw_dir, args.city_map, args.output, args.limit, args.passes)


if __name__ == '__main__':
    main()
