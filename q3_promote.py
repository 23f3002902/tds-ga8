from __future__ import annotations

import math
from datetime import timezone
from typing import Any

from ga8_utils import is_finite_number, is_safe_integer, parse_timestamp, sorted_codes


CANON = __import__('re').compile(r'^[1-9][0-9]*$')


def handle_promote(p: Any) -> tuple[int, dict[str, Any]]:
    if not isinstance(p, dict) or not isinstance(p.get('policy'), dict) or not isinstance(p.get('versions'), list) or not isinstance(p.get('championVersion'), str):
        return 400, {'error': 'INVALID_INPUT'}
    policy=p['policy']; asof=parse_timestamp(p.get('asOf')); champ=p['championVersion']
    req=policy.get('requiredSlices')
    valid_policy=(asof is not None and isinstance(policy.get('datasetDigest'),str) and bool(policy.get('datasetDigest')) and isinstance(policy.get('schemaDigest'),str) and bool(policy.get('schemaDigest'))
      and is_safe_integer(policy.get('maxAgeSeconds')) and is_finite_number(policy.get('accuracyFloor'))
      and is_finite_number(policy.get('maxLatencyMs')) and is_finite_number(policy.get('maxSizeBytes'))
      and is_finite_number(policy.get('minImprovement')) and 0<=policy.get('accuracyFloor',-1)<=1 and 0<=policy.get('minImprovement',-1)<=1
      and isinstance(req,dict) and all(isinstance(k,str) and k and is_finite_number(v) and 0<=v<=1 for k,v in req.items()))
    if not valid_policy: return 400, {'error':'INVALID_INPUT'}
    occ={}
    for v in p['versions']:
        s=v.get('version') if isinstance(v,dict) else None; occ[s]=occ.get(s,0)+1
    failed={}; eligible=[]
    for v in p['versions']:
        key=v.get('version') if isinstance(v,dict) else None
        codes=[]; ev=v.get('evaluation') if isinstance(v,dict) else None
        if not isinstance(key,str) or not CANON.fullmatch(key) or (CANON.fullmatch(key) and int(key)>9007199254740991): codes.append('INVALID_VERSION')
        if occ.get(key)!=1: codes.append('DUPLICATE_VERSION')
        if not isinstance(ev,dict): codes.append('MISSING_EVALUATION')
        if codes:
            pass
        else:
            created=parse_timestamp(ev.get('createdAt'))
            if created is None: codes.append('INVALID_TIMESTAMP')
            else:
                age=(asof-created).total_seconds()
                if age < 0: codes.append('FUTURE_EVALUATION')
                elif age > policy['maxAgeSeconds']: codes.append('STALE_EVALUATION')
            if ev.get('artifactDigest') != v.get('artifactDigest'): codes.append('ARTIFACT_MISMATCH')
            if ev.get('datasetDigest') != policy['datasetDigest']: codes.append('DATASET_MISMATCH')
            if ev.get('schemaDigest') != policy['schemaDigest']: codes.append('SCHEMA_MISMATCH')
            acc,lat,size=ev.get('accuracy'),ev.get('latencyMs'),ev.get('sizeBytes')
            if not all(is_finite_number(x) for x in (acc,lat,size)): codes.append('NON_FINITE')
            if is_finite_number(acc) and not 0<=acc<=1: codes.append('METRIC_RANGE')
            if is_finite_number(acc) and acc < policy['accuracyFloor']: codes.append('ACCURACY_FLOOR')
            if is_finite_number(lat) and lat > policy['maxLatencyMs']: codes.append('LATENCY_LIMIT')
            if is_finite_number(size) and size > policy['maxSizeBytes']: codes.append('SIZE_LIMIT')
            slices=ev.get('slices') if isinstance(ev.get('slices'),dict) else {}
            for name,floor in req.items():
                if name not in slices: codes.append(f'MISSING_SLICE:{name}')
                elif not is_finite_number(slices[name]) or not 0<=slices[name]<=1: codes.append(f'SLICE_RANGE:{name}')
                elif slices[name] < floor: codes.append(f'SLICE_FLOOR:{name}')
        failed[str(key) if key is not None else 'null']=sorted_codes(codes)
        if not codes: eligible.append(v)
    eligible.sort(key=lambda v:(-float(v['evaluation']['accuracy']),float(v['evaluation']['latencyMs']),float(v['evaluation']['sizeBytes']),int(v['version'])))
    champ_obj=next((v for v in eligible if v['version']==champ),None)
    action='block'; selected=None; mutation=None; evidence=None
    if champ_obj:
        winner=eligible[0]
        improvement=round(float(winner['evaluation']['accuracy'])-float(champ_obj['evaluation']['accuracy']),12)
        if winner['version']!=champ and improvement>=float(policy['minImprovement']):
            action='promote'; selected=winner['version']; mutation={'alias':'champion','version':selected}
        else: action='retain'; selected=champ
        evidence=next(v['evaluation'] for v in eligible if v['version']==selected)
    return 200, {'action':action,'championVersion':champ,'selectedVersion':selected,'eligibleVersions':[v['version'] for v in eligible],
                 'failedGates':failed,'aliasMutation':mutation,'evidence':evidence}
