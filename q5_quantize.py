from __future__ import annotations
import copy,hashlib,threading
from typing import Any
from ga8_utils import compact_json,is_finite_number,is_safe_integer,sorted_codes,utf8_key
_F={}; _L=threading.Lock()
def _fp(x): return hashlib.sha256(compact_json(x,sort_keys=True).encode()).hexdigest()
def _inventory(files):
    if not isinstance(files,dict) or any(not isinstance(k,str) or not isinstance(v,str) for k,v in files.items()): return None,None,None
    inv=[{'name':k,'bytes':len(v.encode()),'sha256':hashlib.sha256(v.encode()).hexdigest()} for k,v in sorted(files.items(),key=lambda z:utf8_key(z[0]))]
    total=sum(x['bytes'] for x in inv); digest=hashlib.sha256(compact_json(inv).encode()).hexdigest()
    return inv,total,digest
def _freeze(p):
    fid=p.get('freezeId'); cs=p.get('candidates')
    if not isinstance(fid,str) or not 1<=len(fid)<=128 or not isinstance(cs,list) or not cs: return 400,{'error':'INVALID_INPUT'}
    finger=_fp(p)
    with _L:
        if fid in _F: return (200,copy.deepcopy(_F[fid][1])) if _F[fid][0]==finger else (409,{'error':'FREEZE_ID_CONFLICT'})
    allowed=p.get('allowedUnsupportedReasons'); validbase=isinstance(p.get('calibrationDigest'),str) and bool(p['calibrationDigest']) and isinstance(p.get('tokenizerDigest'),str) and bool(p['tokenizerDigest']) and isinstance(allowed,list) and len(allowed)==len(set(allowed)) and all(isinstance(x,str) and x for x in allowed)
    out=[]
    for c in cs:
        codes=[]; name=c.get('name') if isinstance(c,dict) else None; inv,total,pkg=_inventory(c.get('files') if isinstance(c,dict) else None)
        if not validbase or not isinstance(name,str) or not name or inv is None: codes.append('INVALID_INPUT')
        reason=c.get('unsupportedReason') if isinstance(c,dict) else None
        status='invalid'
        if reason is not None:
            if reason in allowed: status='unsupported'
            else: codes.append('UNALLOWED_UNSUPPORTED_REASON')
        else:
            if c.get('loadable') is not True: codes.append('NOT_LOADABLE')
            if c.get('calibrationDigest')!=p.get('calibrationDigest'): codes.append('CALIBRATION_MISMATCH')
            if c.get('tokenizerDigest')!=p.get('tokenizerDigest'): codes.append('TOKENIZER_MISMATCH')
            if not codes: status='frozen'
        if inv is None: inv=[]; total=pkg=None
        out.append({'name':name,'status':status,'inventory':inv,'totalBytes':total,'packageDigest':pkg,'reasonCodes':sorted_codes(codes)})
    resp={'freezeId':fid,'candidates':sorted(out,key=lambda c:utf8_key(c['name'] or ''))}
    with _L:_F[fid]=(finger,copy.deepcopy(resp))
    return 200,resp
def _select(p):
    fid=p.get('freezeId'); supplied=p.get('candidates'); rows=p.get('rows'); pol=p.get('policy')
    if not isinstance(supplied,list) or not isinstance(rows,list) or not isinstance(pol,dict): return 400,{'error':'INVALID_INPUT'}
    with _L: stored=copy.deepcopy(_F.get(fid))
    frozen=stored[1]['candidates'] if stored else []
    lineage=stored is not None and supplied==frozen
    order=pol.get('candidateOrder'); names=[c.get('name') for c in frozen]
    policy_ok=isinstance(order,list) and len(order)==len(set(order)) and set(order)==set(names) and is_safe_integer(pol.get('maxBytes')) and is_finite_number(pol.get('aggregateFloor')) and 0<=pol['aggregateFloor']<=1 and is_finite_number(pol.get('maxLatencyMs')) and pol['maxLatencyMs']>=0 and isinstance(pol.get('requiredSlices'),dict) and isinstance(p.get('latencies'),dict)
    results=[]
    for c in frozen:
        n=c['name']; codes=[]
        if c['status']!='frozen': codes.append('NOT_FROZEN')
        if not lineage: codes.append('INVALID_LINEAGE')
        if not policy_ok: codes.append('INVALID_POLICY')
        inv,total,pkg=_inventory({x['name']: next((v for k,v in {}.items()),'') for x in []})
        manifest_ok=isinstance(c.get('inventory'),list) and sum(x.get('bytes',0) for x in c['inventory'])==c.get('totalBytes') and hashlib.sha256(compact_json(c['inventory']).encode()).hexdigest()==c.get('packageDigest')
        if not manifest_ok: codes.append('INVALID_MANIFEST')
        predok=bool(rows) and all(isinstance(r,dict) and r.get('label') in {0,1} and isinstance(r.get('slice'),str) and r['slice'] and isinstance(r.get('predictions'),dict) and r['predictions'].get(n) in {0,1} for r in rows)
        agg=None; slices={}
        if not predok: codes.append('INVALID_PREDICTIONS')
        elif policy_ok:
            agg=round(sum(r['label']==r['predictions'][n] for r in rows)/len(rows),12)
            if agg<pol['aggregateFloor']: codes.append('AGGREGATE_FLOOR')
            for s,floor in pol['requiredSlices'].items():
                sr=[r for r in rows if r['slice']==s]
                if not sr: codes.append(f'MISSING_SLICE:{s}'); slices[s]=None
                else:
                    slices[s]=round(sum(r['label']==r['predictions'][n] for r in sr)/len(sr),12)
                    if slices[s]<floor: codes.append(f'SLICE_FLOOR:{s}')
        latency=p.get('latencies',{}).get(n); latency=latency if is_finite_number(latency) and latency>=0 else None
        if policy_ok and c.get('totalBytes') is not None and c['totalBytes']>pol['maxBytes']: codes.append('SIZE_LIMIT')
        if policy_ok and latency is not None and latency>pol['maxLatencyMs']: codes.append('LATENCY_LIMIT')
        results.append({'name':n,'aggregate':agg,'slices':slices,'totalBytes':c.get('totalBytes') if manifest_ok else None,'latencyMs':latency,'admitted':not codes,'reasonCodes':sorted_codes(codes)})
    pos={n:i for i,n in enumerate(order or [])}; results.sort(key=lambda x:(pos.get(x['name'],999999),utf8_key(x['name'])))
    good=[r for r in results if r['admitted']]; good.sort(key=lambda r:(r['totalBytes'],r['latencyMs'],pos.get(r['name'],999999)))
    win=good[0]['name'] if good else None; manifest=next((c for c in frozen if c['name']==win),None)
    return 200,{'freezeId':fid,'selected':win,'results':results,'packageManifest':manifest}
def handle_quantize(p:Any):
    if not isinstance(p,dict): return 400,{'error':'INVALID_INPUT'}
    return _freeze(p) if p.get('phase')=='freeze' else _select(p) if p.get('phase')=='select' else (400,{'error':'INVALID_INPUT'})
