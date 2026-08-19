from __future__ import annotations
import hashlib,json,re
from typing import Any
from ga8_utils import compact_json,is_finite_number,is_safe_integer,sorted_codes,utf8_key
REQ={'README.md','training_manifest.json','evaluation.json','inventory.json','adapter_model.safetensors','adapter_config.json'}
def handle_bundle(p:Any):
    if not isinstance(p,dict) or not isinstance(p.get('policy'),dict) or not isinstance(p.get('files'),dict): return 400,{'error':'INVALID_INPUT'}
    pol=p['policy']; files=p['files']; violations=[]
    slices=pol.get('requiredSlices')
    policy_ok=isinstance(slices,list) and bool(slices) and len(slices)==len(set(slices)) and all(isinstance(x,str) and x for x in slices) and all(isinstance(pol.get(x),str) and pol[x] for x in ['license','intendedUse','limitations'])
    if not policy_ok: violations.append('INVALID_POLICY'); slices=slices if isinstance(slices,list) else []
    for n in sorted(REQ-set(files),key=utf8_key): violations.append(f'MISSING_FILE:{n}')
    for n,v in files.items():
        if not isinstance(n,str) or not isinstance(v,str): violations.append(f'INVALID_FILE:{n}')
    if set(files)-REQ: violations.append('UNTRACKED_FILE')
    if any(k.lower().endswith(('.bin','.pt','.pth','.pkl','.pickle')) for k in files): violations.append('UNSAFE_WEIGHTS')
    recomputed=[{'name':n,'bytes':len(v.encode()),'sha256':hashlib.sha256(v.encode()).hexdigest()} for n,v in sorted(files.items(),key=lambda x:utf8_key(x[0])) if n!='inventory.json' and isinstance(n,str) and isinstance(v,str)]
    invdigest=hashlib.sha256(compact_json(recomputed).encode()).hexdigest()
    try: inv=json.loads(files.get('inventory.json','')); invok=compact_json(inv)==compact_json(recomputed)
    except: invok=False; violations.append('INVALID_JSON:inventory.json')
    if not invok: violations.append('INVENTORY_MISMATCH')
    try: cfg=json.loads(files.get('adapter_config.json','')); cfgok=isinstance(cfg,dict) and is_safe_integer(cfg.get('r'),positive=True) and isinstance(cfg.get('target_modules'),list) and bool(cfg['target_modules']) and len(cfg['target_modules'])==len(set(cfg['target_modules'])) and all(isinstance(x,str) and x for x in cfg['target_modules'])
    except: cfgok=False; violations.append('INVALID_JSON:adapter_config.json')
    if not cfgok: violations.append('INVALID_ADAPTER_CONFIG')
    modeldig=hashlib.sha256(files.get('adapter_model.safetensors','').encode()).hexdigest(); evaldig=hashlib.sha256(files.get('evaluation.json','').encode()).hexdigest()
    try: tm=json.loads(files.get('training_manifest.json','')); tmok=isinstance(tm,dict)
    except: tm={}; tmok=False; violations.append('INVALID_JSON:training_manifest.json')
    if not tmok: violations.append('INVALID_TRAINING_MANIFEST')
    required=['task','baseRevision','datasetDigest','codeDigest','trainingConfigDigest','modelArtifactDigest','evaluationArtifactDigest']
    for k in required:
        if k not in tm: violations.append(f'MISSING_MANIFEST_FIELD:{k}')
    if not isinstance(tm.get('baseRevision'),str) or re.fullmatch(r'[0-9a-f]{40}',tm.get('baseRevision','')) is None: violations.append('MUTABLE_BASE_REVISION')
    for k in ['task','datasetDigest','codeDigest','trainingConfigDigest','modelArtifactDigest','evaluationArtifactDigest']:
        if k in tm and (not isinstance(tm[k],str) or not tm[k]): violations.append('INVALID_TRAINING_MANIFEST')
    if tm.get('modelArtifactDigest')!=modeldig: violations.append('MODEL_ARTIFACT_MISMATCH')
    if tm.get('evaluationArtifactDigest')!=evaldig: violations.append('EVALUATION_DIGEST_MISMATCH')
    try: ev=json.loads(files.get('evaluation.json','')); evok=isinstance(ev,dict)
    except: ev={}; evok=False; violations.append('INVALID_JSON:evaluation.json')
    if not evok: violations.append('INVALID_EVALUATION')
    if ev.get('modelArtifactDigest')!=modeldig: violations.append('EVALUATION_ARTIFACT_MISMATCH')
    if not is_finite_number(ev.get('aggregate')) or not 0<=ev.get('aggregate',-1)<=1: violations.append('INVALID_AGGREGATE')
    es=ev.get('slices') if isinstance(ev.get('slices'),dict) else {}
    for s in slices:
        if s not in es: violations.append(f'MISSING_SLICE:{s}')
        elif not is_finite_number(es[s]) or not 0<=es[s]<=1: violations.append(f'SLICE_RANGE:{s}')
    readme=files.get('README.md',''); markers=re.findall(r'<!-- tds-model-card (.*?) -->',readme,flags=re.S)
    card=None
    if len(markers)!=1:
        violations.append('MODEL_CARD_COUNT')
        if not markers: violations.append('MISSING_MODEL_CARD')
    else:
        try: card=json.loads(markers[0]); assert isinstance(card,dict)
        except: violations.append('INVALID_MODEL_CARD'); card=None
    if card is not None:
        expected={k:tm.get(k) for k in ['task','baseRevision','datasetDigest','modelArtifactDigest']}; expected.update({k:pol[k] for k in ['license','intendedUse','limitations']})
        if any(card.get(k)!=v for k,v in expected.items()): violations.append('MODEL_CARD_MISMATCH')
    violations=sorted_codes(violations)
    return 200,{'decision':'admit' if not violations else 'reject','violations':violations,'inventoryDigest':invdigest}
