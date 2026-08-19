from __future__ import annotations
import copy, hashlib, threading
from typing import Any
from ga8_utils import compact_json, is_finite_number, is_safe_integer, sorted_codes, utf8_key

_F: dict[str, tuple[str, dict[str, Any]]] = {}
_L = threading.Lock()

def _fp(x): return hashlib.sha256(compact_json(x, sort_keys=True).encode("utf-8")).hexdigest()

def _unique_strings(x, nonempty=False):
    return isinstance(x,list) and (not nonempty or bool(x)) and all(isinstance(v,str) and v for v in x) and len(x)==len(set(x))

def _inventory(files):
    if not isinstance(files,dict) or not files or any(not isinstance(k,str) or not k or not isinstance(v,str) for k,v in files.items()):
        return [],None,None
    inv=[]
    for name in sorted(files,key=utf8_key):
        data=files[name].encode("utf-8")
        inv.append({"name":name,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
    total=sum(x["bytes"] for x in inv)
    return inv,total,hashlib.sha256(compact_json(inv).encode("utf-8")).hexdigest()

def _freeze(p):
    fid,cs=p.get("freezeId"),p.get("candidates")
    if not isinstance(fid,str) or not 1<=len(fid)<=128 or not isinstance(cs,list) or not cs:
        return 400,{"error":"INVALID_INPUT"}
    finger=_fp(p)
    with _L:
        if fid in _F:
            return (200,copy.deepcopy(_F[fid][1])) if _F[fid][0]==finger else (409,{"error":"FREEZE_ID_CONFLICT"})
    cal,tok,allowed=p.get("calibrationDigest"),p.get("tokenizerDigest"),p.get("allowedUnsupportedReasons")
    baseok=isinstance(cal,str) and bool(cal) and isinstance(tok,str) and bool(tok)
    allowedok=_unique_strings(allowed)
    names=[c.get("name") if isinstance(c,dict) else None for c in cs]
    namesok=all(isinstance(n,str) and n for n in names) and len(names)==len(set(names))
    out=[]
    for c,n in zip(cs,names):
        codes=[]; inv,total,pkg=_inventory(c.get("files") if isinstance(c,dict) else None)
        if not isinstance(c,dict) or not baseok or not allowedok or not namesok or total is None: codes.append("INVALID_INPUT")
        has_reason=isinstance(c,dict) and "unsupportedReason" in c
        reason=c.get("unsupportedReason") if isinstance(c,dict) else None
        reason_allowed = has_reason and isinstance(reason,str) and bool(reason) and allowedok and reason in allowed
        if has_reason:
            if not isinstance(reason,str) or not reason: codes.append("INVALID_INPUT")
            elif not allowedok or reason not in allowed: codes.append("UNALLOWED_UNSUPPORTED_REASON")
        if isinstance(c,dict) and not reason_allowed:
            if c.get("loadable") is not True: codes.append("NOT_LOADABLE")
            if c.get("calibrationDigest")!=cal: codes.append("CALIBRATION_MISMATCH")
            if c.get("tokenizerDigest")!=tok: codes.append("TOKENIZER_MISMATCH")
        codes=sorted_codes(codes)
        status="invalid" if codes else "unsupported" if has_reason else "frozen"
        out.append({"name":n,"status":status,"inventory":inv,"totalBytes":total,"packageDigest":pkg,"reasonCodes":codes})
    out.sort(key=lambda c:utf8_key(c["name"]) if isinstance(c["name"],str) else b"")
    resp={"freezeId":fid,"candidates":out}
    with _L: _F[fid]=(finger,copy.deepcopy(resp))
    return 200,resp

def _manifest(c):
    inv=c.get("inventory") if isinstance(c,dict) else None
    if not isinstance(inv,list) or not inv: return False,None
    seen=set(); previous=None; total=0
    for x in inv:
        if not isinstance(x,dict) or list(x.keys())!="name bytes sha256".split(): return False,None
        n,b,d=x.get("name"),x.get("bytes"),x.get("sha256")
        if not isinstance(n,str) or not n or n in seen or not is_safe_integer(b) or not isinstance(d,str) or len(d)!=64 or any(ch not in "0123456789abcdef" for ch in d): return False,None
        key=utf8_key(n)
        if previous is not None and previous>key: return False,None
        previous=key;seen.add(n);total+=b
        if total>9007199254740991:return False,None
    pkg=hashlib.sha256(compact_json(inv).encode("utf-8")).hexdigest()
    return (True,total) if total==c.get("totalBytes") and pkg==c.get("packageDigest") else (False,None)

def _policy_ok(pol,names,lats):
    if not isinstance(pol,dict) or not all(isinstance(n,str) and n for n in names): return False
    order,req=pol.get("candidateOrder"),pol.get("requiredSlices")
    return (len(names)==len(set(names)) and _unique_strings(order,True) and set(order)==set(names)
      and is_safe_integer(pol.get("maxBytes")) and is_finite_number(pol.get("aggregateFloor")) and 0<=pol["aggregateFloor"]<=1
      and isinstance(req,dict) and all(isinstance(k,str) and k and is_finite_number(v) and 0<=v<=1 for k,v in req.items())
      and is_finite_number(pol.get("maxLatencyMs")) and pol["maxLatencyMs"]>=0 and isinstance(lats,dict)
      and all(isinstance(k,str) and is_finite_number(v) and v>=0 for k,v in lats.items()))

def _select(p):
    supplied,rows,pol=p.get("candidates"),p.get("rows"),p.get("policy")
    if not isinstance(supplied,list) or not isinstance(rows,list) or not isinstance(pol,dict): return 400,{"error":"INVALID_INPUT"}
    fid=p.get("freezeId")
    with _L: stored=copy.deepcopy(_F.get(fid)) if isinstance(fid,str) else None
    recorded=stored[1]["candidates"] if stored else []; lineage=stored is not None and supplied==recorded
    names=[c.get("name") if isinstance(c,dict) else None for c in supplied]
    lats=p.get("latencies"); policyok=_policy_ok(pol,names,lats)
    req=pol.get("requiredSlices") if isinstance(pol.get("requiredSlices"),dict) else {}
    order=pol.get("candidateOrder") if isinstance(pol.get("candidateOrder"),list) else []
    results=[]
    for c,n in zip(supplied,names):
        codes=[]
        if not isinstance(c,dict) or c.get("status")!="frozen": codes.append("NOT_FROZEN")
        if not lineage: codes.append("INVALID_LINEAGE")
        if not policyok: codes.append("INVALID_POLICY")
        mok,total=_manifest(c)
        if not mok: codes.append("INVALID_MANIFEST")
        predok=bool(rows) and isinstance(n,str) and all(isinstance(r,dict) and not isinstance(r.get("label"),bool) and r.get("label") in {0,1} and isinstance(r.get("slice"),str) and bool(r.get("slice")) and isinstance(r.get("predictions"),dict) and not isinstance(r["predictions"].get(n),bool) and r["predictions"].get(n) in {0,1} for r in rows)
        agg=None;slices={k:None for k in req}
        if not predok: codes.append("INVALID_PREDICTIONS")
        elif policyok:
            agg=round(sum(r["label"]==r["predictions"][n] for r in rows)/len(rows),12)
            if agg<pol["aggregateFloor"]: codes.append("AGGREGATE_FLOOR")
            for s,floor in req.items():
                sr=[r for r in rows if r["slice"]==s]
                if not sr: codes.append(f"MISSING_SLICE:{s}")
                else:
                    slices[s]=round(sum(r["label"]==r["predictions"][n] for r in sr)/len(sr),12)
                    if slices[s]<floor: codes.append(f"SLICE_FLOOR:{s}")
        latency=lats.get(n) if isinstance(lats,dict) and isinstance(n,str) and is_finite_number(lats.get(n)) and lats.get(n)>=0 else None
        if policyok and latency is None: codes.append("INVALID_POLICY")
        if policyok and total is not None and total>pol["maxBytes"]: codes.append("SIZE_LIMIT")
        if policyok and latency is not None and latency>pol["maxLatencyMs"]: codes.append("LATENCY_LIMIT")
        codes=sorted_codes(codes)
        results.append({"name":n,"aggregate":agg,"slices":slices,"totalBytes":total,"latencyMs":latency,"admitted":not codes,"reasonCodes":codes})
    pos={n:i for i,n in enumerate(order) if isinstance(n,str)}
    results.sort(key=lambda x:(pos.get(x["name"],10**9),utf8_key(x["name"]) if isinstance(x["name"],str) else b""))
    good=[r for r in results if r["admitted"]]
    good.sort(key=lambda r:(r["totalBytes"],r["latencyMs"],pos.get(r["name"],10**9)))
    selected=good[0]["name"] if good else None
    manifest=next((c for c in recorded if c.get("name")==selected),None)
    return 200,{"freezeId":fid,"selected":selected,"results":results,"packageManifest":manifest}

def handle_quantize(p:Any):
    if not isinstance(p,dict): return 400,{"error":"INVALID_INPUT"}
    return _freeze(p) if p.get("phase")=="freeze" else _select(p) if p.get("phase")=="select" else (400,{"error":"INVALID_INPUT"})

def reset_freezes_for_tests():
    with _L: _F.clear()
