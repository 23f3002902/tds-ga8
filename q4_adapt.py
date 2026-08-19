from __future__ import annotations
from typing import Any
from ga8_utils import is_finite_number,is_safe_integer,sorted_codes,utf8_key

NAMES=['prompt_only','retrieval','lora','qlora']

def _choose(p):
    pol=p.get('policy'); cs=p.get('candidates')
    if not isinstance(pol,dict) or not isinstance(cs,list): return {'selected':None,'eligible':[],'totalCosts':{n:None for n in NAMES},'reasonCodes':{n:['INVALID_INPUT'] for n in NAMES}}
    by={c.get('name'):c for c in cs if isinstance(c,dict)}
    out={}; costs={}; eligible=[]
    for n in NAMES:
        c=by.get(n); codes=[]
        fields=['quality','latencyMs','memoryMb','oneTimeCost','recurringCost']
        valid=isinstance(c,dict) and all(is_finite_number(c.get(x)) and c[x]>=0 for x in fields) and is_safe_integer(c.get('labeledExamples'))
        try: cost=round(float(c['oneTimeCost'])+int(pol['horizonRequests'])*float(c['recurringCost']),12) if valid and is_safe_integer(pol.get('horizonRequests')) else None
        except: cost=None
        costs[n]=cost
        if not valid: codes.append('INVALID_INPUT')
        else:
            if c.get('available') is not True: codes.append('UNAVAILABLE')
            if c['quality']<pol.get('minQuality',float('inf')): codes.append('QUALITY_FLOOR')
            if pol.get('freshnessRequired') is True and c.get('freshness') is not True: codes.append('FRESHNESS_REQUIRED')
            if c['latencyMs']>pol.get('maxLatencyMs',-1): codes.append('LATENCY_LIMIT')
            if c['memoryMb']>pol.get('maxMemoryMb',-1): codes.append('MEMORY_LIMIT')
            if c['labeledExamples']>pol.get('maxLabeledExamples',-1): codes.append('DATA_LIMIT')
            if cost is None or cost>pol.get('maxTotalCost',-1): codes.append('COST_LIMIT')
        out[n]=sorted_codes(codes)
        if not codes: eligible.append(n)
    return {'selected':eligible[0] if eligible else None,'eligible':eligible,'totalCosts':costs,'reasonCodes':out}

def _repair(p):
    tokens=p.get('tokens'); valid_tokens=isinstance(tokens,list) and bool(tokens)
    if valid_tokens:
        valid_tokens=all(isinstance(t,dict) and is_safe_integer(t.get('id')) and t.get('role') in {'system','user','assistant'} and isinstance(t.get('padding'),bool) and isinstance(t.get('text'),str) for t in tokens)
    labels=[t['id'] if valid_tokens and t['role']=='assistant' and not t['padding'] else -100 for t in tokens] if isinstance(tokens,list) else []
    params=p.get('parameters'); allowed=p.get('allowedTargets'); pcodes=[]; train=[]
    validp=isinstance(params,list) and isinstance(allowed,list) and bool(allowed) and len(allowed)==len(set(allowed)) and all(isinstance(x,str) for x in allowed)
    if validp:
        seen=set()
        for x in params:
            if not isinstance(x,dict) or not isinstance(x.get('name'),str) or x['name'] in seen or not isinstance(x.get('target'),str) or not is_safe_integer(x.get('numel'),positive=True): validp=False; break
            seen.add(x['name'])
            if x['target'] in allowed and (x['name'].endswith('.lora_A.weight') or x['name'].endswith('.lora_B.weight')): train.append(x)
    train.sort(key=lambda x:utf8_key(x['name']))
    files=p.get('artifactFiles'); adapters=sorted(files,key=utf8_key) if isinstance(files,list) else []
    adapterpass=adapters==['adapter_config.json','adapter_model.safetensors']
    ck=p.get('checkpoint'); ckpass=isinstance(ck,dict) and set(['model','optimizer','scheduler','step','rng','dataPosition'])<=set(ck)
    import re
    digestkeys=['datasetDigest','codeDigest','configDigest']
    digests=all(isinstance(p.get(k),str) and re.fullmatch(r'[0-9a-f]{64}',p[k]) is not None for k in digestkeys)
    exp=p.get('expectedDigests'); lineage=digests and isinstance(exp,dict) and all(p.get(k)==exp.get(k) for k in digestkeys)
    base=isinstance(p.get('baseRevision'),str) and re.fullmatch(r'[0-9a-f]{40}',p['baseRevision']) is not None
    tr,ev=p.get('trainRowIds'),p.get('evalRowIds'); isolated=isinstance(tr,list) and isinstance(ev,list) and bool(tr) and bool(ev) and len(tr)==len(set(tr)) and len(ev)==len(set(ev)) and not set(tr)&set(ev)
    batch=all(is_safe_integer(p.get(k),positive=True) for k in ['microBatch','gradientAccumulation','replicas','expectedEffectiveBatch']) and p['microBatch']*p['gradientAccumulation']*p['replicas']==p['expectedEffectiveBatch']
    a,b,tol=p.get('uninterruptedWeights'),p.get('resumedWeights'),p.get('resumeTolerance'); resume=isinstance(a,list) and bool(a) and isinstance(b,list) and len(a)==len(b) and is_finite_number(tol) and tol>=0 and all(is_finite_number(x) and is_finite_number(y) and abs(x-y)<=tol for x,y in zip(a,b))
    codes=[]
    if not valid_tokens: codes.append('INVALID_TOKEN')
    if p.get('templateApplications')!=1: codes.append('CHAT_TEMPLATE_COUNT')
    if not validp or not train: codes.append('INVALID_PARAMETER')
    if p.get('inferenceMode') is not False: codes.append('INFERENCE_MODE')
    if not adapterpass: codes.append('ADAPTER_FILE_SET')
    if ck is not None and not ckpass: codes.append('INCOMPLETE_CHECKPOINT')
    if not base: codes.append('MUTABLE_BASE_REVISION')
    if not lineage: codes.append('LINEAGE_MISMATCH')
    if not batch: codes.append('EFFECTIVE_BATCH_MISMATCH')
    if not isolated: codes.append('EVAL_LEAKAGE')
    if p.get('dropoutActiveDuringEval') is not False: codes.append('EVAL_DROPOUT_ACTIVE')
    if not resume: codes.append('RESUME_DIVERGENCE')
    return {'labels':labels,'templatePass':p.get('templateApplications')==1,'trainableParams':[x['name'] for x in train],
      'trainableCount':sum(x['numel'] for x in train),'peftConfigPass':validp and bool(train) and p.get('inferenceMode') is False,
      'adapterFiles':adapters,'checkpointComplete':ckpass,'lineagePass':lineage and base and batch,'evalIsolated':isolated,
      'evaluationDeterministic':p.get('dropoutActiveDuringEval') is False,'resumePass':resume,'reasonCodes':sorted_codes(codes)}

def handle_adapt(p:Any):
    if not isinstance(p,dict) or p.get('operation') not in {'choose','repair'}: return 400,{'error':'INVALID_INPUT'}
    return 200,_choose(p) if p['operation']=='choose' else _repair(p)
