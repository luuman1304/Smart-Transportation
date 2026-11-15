"""Isolated author-code JPT reproduction, training-only High+Delivered holdout."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import natsort

from .benchmark import objects
from .baselines import evaluate, validate_sequence

ROOT = Path(__file__).resolve().parents[2]
JPT = ROOT / 'third_party/jpt-amz'
SCORER = ROOT / 'third_party/rc-cli/scoring/score.py'


def load_scorer():
    spec = importlib.util.spec_from_file_location('mit_score', SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_score(actual, proposed, costs):
    validate_sequence(proposed, actual, actual[0])
    # Official normalize_matrix shallow-copies then mutates nested cost rows.
    # Protect raw seconds used by the operational evaluator and other methods.
    return float(load_scorer().score(actual + actual[:1], proposed + proposed[:1], copy.deepcopy(costs)))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2))


def command(args, cwd, log):
    with log.open('w') as handle:
        subprocess.run([str(a) for a in args], cwd=cwd, stdout=handle,
                       stderr=subprocess.STDOUT, check=True,
                       env={**os.environ, 'PYTHONHASHSEED': '0', 'TZ': 'UTC'})


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    source = ROOT / 'data/raw/amazon/almrrc2021-data-training/model_build_inputs'
    routes = dict(objects(source / 'route_data.json'))
    actual = dict(objects(source / 'actual_sequences.json'))
    selected = []
    selected_packages = {}
    for rid, package in objects(source / 'package_data.json'):
        failed = any(p.get('scan_status') == 'DELIVERY_ATTEMPTED'
                     for stop in package.values() for p in stop.values())
        if routes[rid]['route_score'] == 'High' and not failed:
            selected.append(rid)
            for stop in package.values():
                for p in stop.values():
                    p.pop('scan_status', None)
            selected_packages[rid] = package
    selected = sorted(selected)
    if len(selected) != 1107:
        raise ValueError(f'author High+Delivered rule gave {len(selected)}, expected 1107')
    held = set(selected)
    fit = [rid for rid in routes if rid not in held]
    assert not set(fit) & held and len(fit) == 5005
    dump(out / 'split.json', dict(fit=fit, holdout=selected,
         rule='High and no DELIVERY_ATTEMPTED, matching author markFailedRoutes',
         note='Training-only; previously inspected in aggregate; not pristine final test'))
    dump(out / 'fit_routes.json', {r: routes[r] for r in fit})
    dump(out / 'fit_sequences.json', {r: actual[r] for r in fit})
    # Scoring labels live outside files passed to author apply scripts.
    dump(out / 'holdout_actual.json', {r: actual[r] for r in selected})
    dump(out / 'holdout_routes.json', {r: {k: v for k, v in routes[r].items()
                                         if k != 'route_score'} for r in selected})
    dump(out / 'holdout_packages.json', {r: selected_packages[r] for r in selected})
    del selected_packages
    with (out / 'holdout_times.json').open('w') as handle:
        handle.write('{')
        first = True
        for rid, costs in objects(source / 'travel_times.json'):
            if rid in held:
                handle.write(('' if first else ',') + json.dumps(rid) + ':' + json.dumps(costs))
                first = False
        handle.write('}')
    provenance = dict(python=sys.version, numpy=np.__version__, natsort=natsort.__version__,
                      platform=platform.platform(), scorer_sha256=fingerprint(SCORER),
                      jpt_commit=subprocess.check_output(['git','-C',str(JPT),'rev-parse','HEAD'],text=True).strip(),
                      scorer_commit=subprocess.check_output(['git','-C',str(SCORER.parents[1]),'rev-parse','HEAD'],text=True).strip(),
                      sha256={str(p): fingerprint(p) for p in [source/'route_data.json',
                         source/'actual_sequences.json', source/'package_data.json', source/'travel_times.json']})
    dump(out / 'provenance.json', provenance)
    print('Prepared 5005 fit / 1107 holdout routes', flush=True)


def build(out):
    command([sys.executable,JPT/'scripts/analyze_zone_id_order_main.py',
             '--r_file','fit_routes.json','--s_file','fit_sequences.json','--out_file','model.json'], out, out/'build.log')
    for variant in (1, 2):
        args = [sys.executable,JPT/'scripts/build_TSPLIB.py',
                '--r_file','holdout_routes.json','--t_file','holdout_times.json',
                '--p_file','holdout_packages.json','--z_file','model.json',
                '--br_file','fit_routes.json','-noPrune','-noPruneFailed',
                '-zoneNeighborTrans','-superNeighborTrans']
        if variant == 2:
            args += ['-superPred','-trans']
        args += [f'TSPLIB_{variant}']
        command(args, out, out/f'build_variant_{variant}.log')
        if len(list((out/f'TSPLIB_{variant}').glob('*.ctsptw'))) != 1107:
            raise ValueError('incomplete TSPLIB conversion')
        print(f'Built variant {variant}', flush=True)


def solve(out, seconds, workers):
    dump(out/'solver_config.json', dict(seconds_per_variant=seconds, variants=2,
         workers=workers, seed=1, ctsp_transform=True, time_windows=False,
         merge='choose variant1 if length1 <= floor(1.01 * length2), else variant2',
         deviation='fixed per-route budget replaces competition-wide dynamic budget'))
    for variant in (1, 2):
        (out/f'tours_{variant}').mkdir(exist_ok=True)
        (out/f'params_{variant}').mkdir(exist_ok=True)
        instances = sorted((out/f'TSPLIB_{variant}').glob('*.ctsptw'))
        def one(instance):
            name = instance.stem
            par = out/f'params_{variant}'/f'{name}.par'
            par.write_text(f'PROBLEM_FILE = TSPLIB_{variant}/{name}.ctsptw\nCTSP_TRANSFORM = YES\nSEED = 1\nTIME_LIMIT = {seconds}\nTRACE_LEVEL = 0\nTOUR_FILE = tours_{variant}/{name}.tour\n')
            begin = time.perf_counter()
            command([JPT/'LKH-AMZ/LKH', par.relative_to(out)],out,out/f'params_{variant}'/f'{name}.log')
            if not (out/f'tours_{variant}'/f'{name}.tour').exists():
                raise ValueError(f'no solution for {name}')
            return name,time.perf_counter()-begin
        timings = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i,(name,seconds_used) in enumerate(pool.map(one,instances),1):
                timings[name] = seconds_used
                if i % 100 == 0:
                    print(f'Variant {variant}: {i}/1107',flush=True)
        dump(out/f'timing_{variant}.json',timings)


def read_tour(path):
    text = path.read_text()
    match = re.search(r'COMMENT : Cost = (\d+)_(\d+)', text)
    if match is None:
        raise ValueError(f'invalid tour header: {path}')
    nodes = text.split('TOUR_SECTION\n',1)[1].split('-1',1)[0].split()
    return int(match[2]), nodes


def score(out):
    mapping = json.loads((out/'TSPLIB_1/tsp2amz.json').read_text())
    mapping2 = json.loads((out/'TSPLIB_2/tsp2amz.json').read_text())
    assert mapping == mapping2
    predictions = {}
    choices = {}
    for name, entry in mapping.items():
        length1,nodes1 = read_tour(out/'tours_1'/f'{name}.tour')
        length2,nodes2 = read_tour(out/'tours_2'/f'{name}.tour')
        take1 = length1 <= (101 * length2)//100
        nodes = nodes1 if take1 else nodes2
        predictions[entry['route_id']] = [entry['stops'][s] for s in nodes]
        choices[entry['route_id']] = 1 if take1 else 2
    dump(out/'jpt_sequences.json', predictions)
    dump(out/'variant_choices.json', choices)
    actual = json.loads((out/'holdout_actual.json').read_text())
    routes = json.loads((out/'holdout_routes.json').read_text())
    packages = json.loads((out/'holdout_packages.json').read_text())
    assert set(predictions) == set(actual) == set(routes)
    old = {}
    with (ROOT/'outputs/training_baselines_v1/sequences.jsonl').open() as handle:
        for line in handle:
            item = json.loads(line)
            if item['route_id'] in actual and item['method'] != 'observed':
                old[item['route_id'],item['method']] = item['sequence']
    rows = []
    with (out/'results.csv').open('w',newline='') as handle:
        writer = None
        for i,(rid,costs) in enumerate(objects(out/'holdout_times.json'),1):
            ranks = actual[rid]['actual']
            reference = sorted(ranks,key=ranks.get)
            methods = {'jpt_amz': predictions[rid], 'observed':reference,
                       **{m:old[rid,m] for m in ('nearest_neighbor','nearest_neighbor_2opt')}}
            for method,sequence in methods.items():
                metrics = evaluate(sequence,routes[rid],packages[rid],costs)
                value = official_score(reference,sequence,costs)
                if method == 'observed' and value != 0:
                    raise ValueError('official self score must be zero')
                row = dict(route_id=rid,method=method,official_score=value,**metrics)
                if writer is None:
                    writer = csv.DictWriter(handle,fieldnames=list(row)); writer.writeheader()
                writer.writerow(row); rows.append(row)
            if i % 100 == 0:
                handle.flush()
                print(f'Scored {i}/1107 with original MIT scorer',flush=True)
    summary = {}
    for method in methods:
        subset = [r for r in rows if r['method']==method]
        summary[method] = {'routes':len(subset), **{
            'mean_'+k:float(np.mean([r[k] for r in subset]))
            for k in subset[0] if k not in ('route_id','method')}}
    paired = {m:{r['route_id']:r['official_score'] for r in rows if r['method']==m}
              for m in methods}
    rng = np.random.default_rng(42)
    differences = np.array([paired['jpt_amz'][r]-paired['nearest_neighbor_2opt'][r] for r in sorted(actual)])
    boot = [float(np.mean(rng.choice(differences,size=len(differences),replace=True))) for _ in range(10000)]
    summary['paired_vs_nn2opt'] = dict(mean_difference=float(np.mean(differences)),
        ci95=list(np.quantile(boot,[0.025,0.975])), bootstrap_seed=42, bootstrap_replicates=10000,
        fraction_improved=float(np.mean(differences<0)))
    summary['model_gate'] = 'NOT PASSED: reproducing prior method is not beating prior method; single solver seed only'
    dump(out/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/jpt_reproduction_v1'))
    parser.add_argument('--stage',choices=['prepare','build','solve','score','all'],default='all')
    parser.add_argument('--seconds',type=float,default=1)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    if args.seconds <= 0 or args.workers < 1:
        parser.error('positive seconds/workers required')
    out=args.output.resolve()
    for stage,fn in [('prepare',prepare),('build',build),('solve',lambda p:solve(p,args.seconds,args.workers)),('score',score)]:
        if args.stage in (stage,'all'):
            print('Stage: '+stage,flush=True); fn(out)


if __name__ == '__main__':
    main()
