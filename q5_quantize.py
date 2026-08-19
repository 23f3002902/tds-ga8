from __future__ import annotations
import copy,hashlib,threading
from typing import Any
from ga8_utils import compact_json,is_finite_number,is_safe_integer,sorted_codes,utf8_key
_F={}; _L=threading.Lock()
def _fp(x): return hashlib.sha256(compact_json(x,sort_keys=True).encode()).hexdigest()
def _inventory(files):
    if not isinstance(files,dict) or not files or any(not isinstance(k,str) or not k or not isinstance(v,str) for k,v in files.items()): return [],None,None
    inv=[{'name':k,'bytes':len(v.encode()),'sha256':hashlib.sha256(v.encode()).hexdigest()} for k,v in sorted(files.items(),key=lambda z:utf8_key(z[0]))]
    total=sum(x['bytes'] for x in inv); return inv,total,hashlib.sha256(compact_json(inv).encode()).hexdigest()
def _freeze(p):
    fid=p.get('freezeId'); cs=p.get('candidates'); allowed=p.get('allowedUnsupportedReasons')
    if not isinstance(fid,str) or not 1<=len(fid)<=128 or not isinstance(cs,list) or not cs: return 400,{'error':'INVALID_INPUT'}
    finger=_fp(p)
    with _L:
        if fid in _F: return (200,copy.deepcopy(_F[fid][1])) if _F[fid][0]==finger else (409,{'error':'FREEZE_ID_CONFLICT'})
    baseok=isinstance(p.get('calibrationDigest'),str) and bool(p['calibrationDigest']) and isinstance(p.get('tokenizerDigest'),str) and bool(p['tokenizerDigest'])
    allowedok=isinstance(allowed,list) and len(allowed)==len(set(allowed)) and all(isinstance(x,str) and x for x in allowed)
    names=[c.get('name') if isinstance(c,dict) else None for c in cs]
    namesok=all(isinstance(x,str) and x for x in names) and len(names)==len(set(names))
    out=[]
    for c,name in zip(cs,names):
        codes=[]; inv,total,pkg=_inventory(c.get('files') if isinstance(c,dict) else None)
        if not baseok or not allowedok or not namesok or not isinstance(c,dict) or total is None: codes.append('INVALID_INPUT')
        reason=c.get('unsupportedReason') if isinstance(c,dict) else None
        if reason is not None:
            if not isinstance(reason,str) or not reason: codes.append('INVALID_INPUT')
            elif not allowedok or reason not in allowed: codes.append('UNALLOWED_UNSUPPORTED_REASON')
        else:
            if c.get('loadable') is not True: codes.append('NOT_LOADABLE')
            if c.get('calibrationDigest')!=p.get('calibrationDigest'): codes.append('CALIBRATION_MISMATCH')
            if c.get('tokenizerDigest')!=p.get('tokenizerDigest'): codes.append('TOKENIZER_MISMATCH')
        codes=sorted_codes(codes); status='invalid' if codes else ('unsupported' if reason is not None else 'frozen')
        out.append({'name':name,'status':status,'inventory':inv,'totalBytes':total,'packageDigest':pkg,'reasonCodes':codes})
    resp={'freezeId':fid,'candidates':sorted(out,key=lambda c:utf8_key(c['name']) if isinstance(c['name'],str) else b'')}
    with _L:_F[fid]=(finger,copy.deepcopy(resp))
    return 200,resp
def _manifest(c):
    inv=c.get('inventory') if isinstance(c,dict) else None
    if not isinstance(inv,list) or any(not isinstance(x,dict) or list(x)!=['name','bytes','sha256'] or not isinstance(x['name'],str) or not is_safe_integer(x['bytes']) or not isinstance(x['sha256'],str) for x in inv): return False,None
    if inv!=sorted(inv,key=lambda x:utf8_key(x['name'])) or len({x['name'] for x in inv})!=len(inv): return False,None
    total=sum(x['bytes'] for x in inv); pkg=hashlib.sha256(compact_json(inv).encode()).hexdigest()
    return total==c.get('totalBytes') and pkg==c.get('packageDigest'),total
def _select(p):
    supplied=p.get('candidates'); rows=p.get('rows'); pol=p.get('policy')
    if not isinstance(supplied,list) or not isinstance(rows,list) or not isinstance(pol,dict): return 400,{'error':'INVALID_INPUT'}
    fid=p.get('freezeId')
    with _L: stored=copy.deepcopy(_F.get(fid))
    recorded=stored[1]['candidates'] if stored else []; lineage=stored is not None and supplied==recorded
    names=[c.get('name') if isinstance(c,dict) else None for c in supplied]
    order=pol.get('candidateOrder'); lats=p.get('latencies'); req=pol.get('requiredSlices')
    policy_ok=(isinstance(order,list) and len(order)==len(set(order)) and set(order)==set(names) and len(names)==len(set(names)) and all(isinstance(x,str) and x for x in names)
      and is_safe_integer(pol.get('maxBytes')) and is_finite_number(pol.get('aggregateFloor')) and 0<=pol['aggregateFloor']<=1
      and is_finite_number(pol.get('maxLatencyMs')) and pol['maxLatencyMs']>=0 and isinstance(req,dict)
      and all(isinstance(k,str) and k and is_finite_number(v) and 0<=v<=1 for k,v in req.items()) and isinstance(lats,dict))
    results=[]
    for c,n in zip(supplied,names):
        codes=[]
        if not isinstance(c,dict) or c.get('status')!='frozen': codes.append('NOT_FROZEN')
        if not lineage: codes.append('INVALID_LINEAGE')
        if not policy_ok: codes.append('INVALID_POLICY')
        mok,total=_manifest(c)
        if not mok: codes.append('INVALID_MANIFEST'); total=None
        predok=bool(rows) and isinstance(n,str) and all(isinstance(r,dict) and not isinstance(r.get('label'),bool) and r.get('label') in {0,1} and isinstance(r.get('slice'),str) and r['slice'] and isinstance(r.get('predictions'),dict) and not isinstance(r['predictions'].get(n),bool) and r['predictions'].get(n) in {0,1} for r in rows)
        agg=None;slices={k:None for k in req} if isinstance(req,dict) else {}
        if not predok: codes.append('INVALID_PREDICTIONS')
        elif policy_ok:
            agg=round(sum(r['label']==r['predictions'][n] for r in rows)/len(rows),12)
            if agg<pol['aggregateFloor']: codes.append('AGGREGATE_FLOOR')
            for s,floor in req.items():
                sr=[r for r in rows if r['slice']==s]
                if not sr: codes.append(f'MISSING_SLICE:{s}')
                else:
                    slices[s]=round(sum(r['label']==r['predictions'][n] for r in sr)/len(sr),12)
                    if slices[s]<floor: codes.append(f'SLICE_FLOOR:{s}')
        latency=lats.get(n) if isinstance(lats,dict) else None
        if not is_finite_number(latency) or latency<0: latency=None
        if policy_ok and total is not None and total>pol['maxBytes']: codes.append('SIZE_LIMIT')
        if policy_ok and latency is not None and latency>pol['maxLatencyMs']: codes.append('LATENCY_LIMIT')
        if policy_ok and latency is None: codes.append('INVALID_POLICY')
        codes=sorted_codes(codes);results.append({'name':n,'aggregate':agg,'slices':slices,'totalBytes':total,'latencyMs':latency,'admitted':not codes,'reasonCodes':codes})
    pos={n:i for i,n in enumerate(order or [])};results.sort(key=lambda x:(pos.get(x['name'],10**9),utf8_key(x['name']) if isinstance(x['name'],str) else b''))
    good=[r for r in results if r['admitted']];good.sort(key=lambda r:(r['totalBytes'],r['latencyMs'],pos.get(r['name'],10**9)))
    win=good[0]['name'] if good else None;manifest=next((c for c in recorded if c.get('name')==win),None)
    return 200,{'freezeId':fid,'selected':win,'results':results,'packageManifest':manifest}
def handle_quantize(p:Any):
    if not isinstance(p,dict): return 400,{'error':'INVALID_INPUT'}
    return _freeze(p) if p.get('phase')=='freeze' else _select(p) if p.get('phase')=='select' else (400,{'error':'INVALID_INPUT'})
