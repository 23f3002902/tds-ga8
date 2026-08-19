from __future__ import annotations
import copy,hashlib,threading
from typing import Any
from ga8_utils import compact_json,is_safe_integer
NODES=['verify_data','prepare','train','evaluate','register','publish']
REQ=['generation','checksum','canonicalData','prepareCode','prepareConfig','trainCode','trainConfig','runtime','evaluateCode','evaluateConfig','schemaDigest','publishConfig']
_S={};_CACHE={};_L=threading.Lock()
def h(a):return hashlib.sha256(compact_json(a).encode()).hexdigest()
def keys(inp):
    deps={}; ks={}; art={}
    arrays={
      'verify_data':[inp['generation'],inp['checksum']],
      'prepare':[inp['canonicalData'],inp['prepareCode'],inp['prepareConfig']],
      'train':[None,inp['trainCode'],inp['trainConfig'],inp['runtime']],
      'evaluate':[None,inp['canonicalData'],inp['evaluateCode'],inp['evaluateConfig']],
      'register':[None,inp['schemaDigest']], 'publish':[None,inp['publishConfig']]}
    names={
      'verify_data':['generation','checksum'],'prepare':['canonicalData','prepareCode','prepareConfig'],
      'train':['prepareArtifact','trainCode','trainConfig','runtime'],'evaluate':['trainArtifact','canonicalData','evaluateCode','evaluateConfig'],
      'register':['evaluateArtifact','schemaDigest'],'publish':['registerArtifact','publishConfig']}
    for i,n in enumerate(NODES):
        if i >= 2:
            parent=NODES[i-1]; arrays[n][0]=art.get(parent)
        ks[n]=h(arrays[n]) if all(x is not None for x in arrays[n]) else None
        if ks[n] in _CACHE: art[n]=_CACHE[ks[n]]['artifact']
        deps[n]={k:v for k,v in zip(names[n],arrays[n])};deps[n]['cacheKey']=ks[n]
    return ks,deps,art
def handle_pipeline(p:Any):
    if not isinstance(p,dict) or not isinstance(p.get('session'),str) or not p['session'] or not is_safe_integer(p.get('revision'),positive=True) or not isinstance(p.get('inputs'),dict) or not isinstance(p.get('events'),list) or any(not isinstance(p['inputs'].get(k),str) or not p['inputs'][k] for k in REQ):
        return 409,{'error':'INVALID_REQUEST'}
    session,rev,inp=p['session'],p['revision'],p['inputs']
    with _L:
      st=copy.deepcopy(_S.get(session,{'revision':rev,'inputs':copy.deepcopy(inp),'states':{},'events':{}})); cachecopy=copy.deepcopy(_CACHE)
      if rev<st['revision']: pass
      elif rev==st['revision'] and inp!=st['inputs']: return 409,{'error':'REVISION_CONFLICT'}
      elif rev>st['revision']: st={'revision':rev,'inputs':copy.deepcopy(inp),'states':{},'events':st['events']}
      accepted=[];ignored=[]
      for e in p['events']:
        if not isinstance(e,dict) or set(e)!=set(['eventId','revision','node','attempt','status','key','artifactDigest','receiptId']): return 409,{'error':'INVALID_EVENT'}
        eid=e['eventId']; canon=compact_json(e)
        if not isinstance(eid,str) or not eid: return 409,{'error':'INVALID_EVENT'}
        if eid in st['events']:
            if st['events'][eid]!=canon: return 409,{'error':'EVENT_ID_CONFLICT'}
            ignored.append(eid);continue
        ks,deps,arts=keys(st['inputs']); n=e.get('node'); status=e.get('status')
        valid=is_safe_integer(e.get('attempt'),positive=True) and status in {'started','succeeded','retryable_failed','terminal_failed'} and n in NODES
        valid=valid and ((status=='succeeded' and isinstance(e.get('artifactDigest'),str) and bool(e['artifactDigest'])) or (status!='succeeded' and e.get('artifactDigest') is None))
        valid=valid and ((status=='succeeded' and n in {'register','publish'} and e.get('receiptId')==f'receipt:{n}:{e.get("key")}') or ((status!='succeeded' or n not in {'register','publish'}) and e.get('receiptId') is None))
        if not valid or e['revision']!=st['revision'] or e.get('key')!=ks.get(n) or ks.get(n) is None: ignored.append(eid);continue
        prev=st['states'].get(n); attempt=e['attempt']
        if prev is None:
            if status!='started' or attempt!=1: ignored.append(eid);continue
        elif prev['status']=='started':
            if attempt<prev['attempt']: ignored.append(eid);continue
            if attempt!=prev['attempt'] or status not in {'succeeded','retryable_failed','terminal_failed'}: return 409,{'error':'STATUS_CONFLICT'}
        elif prev['status']=='retryable_failed':
            if attempt<prev['attempt']+1: ignored.append(eid);continue
            if status!='started' or attempt!=prev['attempt']+1:return 409,{'error':'STATUS_CONFLICT'}
        elif prev['status']=='succeeded':
            if status=='succeeded' and e['artifactDigest']!=prev['artifactDigest']:return 409,{'error':'EVIDENCE_CONFLICT'}
            return 409,{'error':'STATUS_CONFLICT'}
        else:return 409,{'error':'STATUS_CONFLICT'}
        st['states'][n]={'status':status,'attempt':attempt,'eventId':eid,'artifactDigest':e.get('artifactDigest')};st['events'][eid]=canon;accepted.append(eid)
        if status=='succeeded':_CACHE[e['key']]={'artifact':e['artifactDigest'],'eventId':eid}
      ks,deps,arts=keys(st['inputs']); nodes=[]; upstream=None
      for n in NODES:
        state=st['states'].get(n); trig=[]
        if upstream=='terminal': action,reason='block','UPSTREAM_TERMINAL'
        elif upstream: action,reason='block','UPSTREAM_PENDING'
        elif ks[n] in _CACHE: action,reason='reuse','CACHE_HIT';trig=[_CACHE[ks[n]]['eventId']]
        elif state and state['status']=='started': action,reason='block','RUNNING';trig=[state['eventId']];upstream='pending'
        elif state and state['status']=='terminal_failed': action,reason='block','TERMINAL_FAILURE';trig=[state['eventId']];upstream='terminal'
        elif state and state['status']=='retryable_failed': action,reason='rerun','RETRYABLE_FAILURE';trig=[state['eventId']];upstream='pending'
        else: action,reason='rerun','CACHE_MISS';upstream='pending'
        nodes.append({'node':n,'action':action,'reasonCodes':[reason],'dependencyDigests':deps[n],'triggeringEventIds':trig})
      _S[session]=st
    return 200,{'revision':st['revision'],'acceptedEventIds':accepted,'ignoredEventIds':ignored,'nodes':nodes}
