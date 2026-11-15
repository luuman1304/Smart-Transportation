"""Fit-only zone preferences and guarded directed repair; no scorer dependency."""
from itertools import combinations
import math

from .baselines import evaluate, travel_time, validate_sequence


def zone_blocks(sequence, route):
    blocks=[]
    seen=set()
    for stop in sequence[1:]:
        zone=route['stops'][stop].get('zone_id')
        if not isinstance(zone,str) or not zone:
            return None
        if blocks and blocks[-1][0]==zone:
            blocks[-1][1].append(stop)
        else:
            if zone in seen:
                return None
            seen.add(zone); blocks.append((zone,[stop]))
    return blocks


def history_record(rid, route, actual):
    order=sorted(actual['actual'],key=actual['actual'].get)
    occurrences={}
    previous=None
    block=0
    for stop in order:
        if route['stops'][stop]['type']=='Station':
            continue
        z=route['stops'][stop].get('zone_id')
        if z!=previous:
            block+=1
        if isinstance(z,str) and z:
            occurrences.setdefault(z,set()).add(block)
        previous=z
    # Every pair involving a revisited zone is excluded.
    positions={z:next(iter(v)) for z,v in occurrences.items() if len(v)==1}
    return dict(route_id=rid,station=route['station_code'],positions=positions)


def preferences(route, zones, history, k):
    target=set(zones)
    eligible=[]
    for h in history:
        if h['station']!=route['station_code']:
            continue
        shared=target & set(h['positions'])
        if len(shared)>=2:
            similarity=len(shared)/len(target | set(h['positions']))
            eligible.append((-similarity,h['route_id'],h))
    selected=[h for _,_,h in sorted(eligible,key=lambda x:(x[0],x[1]))[:k]]
    result=[]
    for a,b in combinations(sorted(target),2):
        votes=[h['positions'][a]<h['positions'][b] for h in selected
               if a in h['positions'] and b in h['positions']]
        if len(votes)<2:
            continue
        # Beta(1,1) smoothing. Confidence is |2p-1|, ties add no constraint.
        confidence=abs(2*sum(votes)-len(votes))/(len(votes)+2)
        if confidence:
            before,after=(a,b) if 2*sum(votes)>len(votes) else (b,a)
            result.append((before,after,confidence))
    return result


def hierarchy_groups(path, mapping):
    """Read explicit JPT hierarchy rather than guessing from zone strings."""
    levels={}; active=None
    for line in path.read_text().splitlines():
        line=line.strip()
        if line in ('GTSP_SET_SECTION','SUPER_GTSP_SET_SECTION','SUPER_SUPER_GTSP_SET_SECTION'):
            active=line; levels[active]={}; continue
        if not line or not line[0].isdigit():
            active=None; continue
        if active:
            values=line.split()
            if values[-1]!='-1':
                raise ValueError('invalid hierarchy row')
            levels[active][values[0]]=values[1:-1]
    prev={n:{s} for n,s in mapping['stops'].items()}
    groups=[]
    for section in ('GTSP_SET_SECTION','SUPER_GTSP_SET_SECTION','SUPER_SUPER_GTSP_SET_SECTION'):
        current={key:set().union(*(prev[v] for v in values))
                 for key,values in levels.get(section,{}).items()}
        if not current:
            raise ValueError('missing JPT hierarchy')
        groups.extend(current.values()); prev=current
    return groups


def contiguous(sequence, groups):
    pos={s:i for i,s in enumerate(sequence)}
    return all(max(pos[s] for s in g)-min(pos[s] for s in g)+1==len(g) for g in groups)


def moves(sequence,route,within):
    blocks=zone_blocks(sequence,route)
    if blocks is None:
        return
    chunks=[b[1] for b in blocks]
    def flatten(parts):
        return [sequence[0]]+[s for part in parts for s in part]
    for i in range(len(chunks)):
        for j in range(i+1,len(chunks)):
            candidate=list(chunks); candidate[i],candidate[j]=candidate[j],candidate[i]
            yield flatten(candidate)
    for i in range(len(chunks)):
        for j in range(len(chunks)):
            if i!=j:
                candidate=list(chunks); block=candidate.pop(i); candidate.insert(j,block)
                yield flatten(candidate)
    if within:
        start=1
        for chunk in chunks:
            for i in range(start,start+len(chunk)-1):
                for j in range(i+1,start+len(chunk)):
                    yield sequence[:i]+list(reversed(sequence[i:j+1]))+sequence[j+1:]
            start+=len(chunk)


def penalty(sequence,route,prior):
    pos={}
    for i,s in enumerate(sequence[1:]):
        pos.setdefault(route['stops'][s].get('zone_id'),i)
    total=sum(w for a,b,w in prior)
    return sum(w for a,b,w in prior if pos[a]>pos[b])/total if total else 0.0


def guarded(metric,base):
    return (metric['travel_seconds']<=base['travel_seconds']*1.05+1e-7 and
            all(metric[k]<=base[k]+1e-7 for k in
                ('duration_seconds','package_lateness_seconds','late_packages')))


def repair(sequence,route,packages,costs,prior,groups,weight,within,
           passes=2,move_cap=1200):
    base=evaluate(sequence,route,packages,costs)
    if zone_blocks(sequence,route) is None or not contiguous(sequence,groups):
        return list(sequence),base,dict(reason='ambiguous_or_noncontiguous',moves=0,accepted=0)
    if weight and not prior:
        return list(sequence),base,dict(reason='unsupported',moves=0,accepted=0)
    # Input-only scale, independent of incumbent and all observed sequences.
    n=len(costs)
    scale=max(1.0,sum(v for a,row in costs.items() for b,v in row.items() if a!=b)/max(1,n-1))
    current=list(sequence); metric=base; examined=accepted=0
    for _ in range(passes):
        best=metric['travel_seconds']/scale+weight*penalty(current,route,prior)
        best_seq=None; best_metric=None
        for index,candidate in enumerate(moves(current,route,within)):
            if index>=move_cap:
                break
            examined+=1
            if not contiguous(candidate,groups):
                continue
            travel=travel_time(candidate,costs)
            objective=travel/scale+weight*penalty(candidate,route,prior)
            if objective>=best-1e-12 or travel>base['travel_seconds']*1.05+1e-7:
                continue
            candidate_metric=evaluate(candidate,route,packages,costs)
            if guarded(candidate_metric,base):
                best=objective; best_seq=candidate; best_metric=candidate_metric
        if best_seq is None:
            break
        current=best_seq; metric=best_metric; accepted+=1
    validate_sequence(current,route['stops'],sequence[0])
    return current,metric,dict(reason='searched',moves=examined,accepted=accepted)
