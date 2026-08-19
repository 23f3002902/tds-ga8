from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from q1_corpus import build_corpus
from q2_bqml import handle_bqml
from q3_promote import handle_promote
from q4_adapt import handle_adapt
from q5_quantize import handle_quantize
from q6_pipeline import handle_pipeline
from q7_bundle import handle_bundle


app = FastAPI(title="TDS GA8", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def read_json(request: Request):
    try:
        return await request.json()
    except Exception:
        return None


@app.get("/")
def root() -> dict[str, object]:
    return {"ok": True, "assignment": "tds-2026-05-ga8"}


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/build-corpus")
async def corpus_endpoint(request: Request):
    payload = await read_json(request)
    result = build_corpus(payload)
    if result is None:
        return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)
    return JSONResponse(result)


@app.post("/bqml")
async def bqml_endpoint(request: Request):
    payload = await read_json(request)
    status, result = handle_bqml(payload)
    return JSONResponse(result, status_code=status)

async def dispatch(request, handler):
    status, result = handler(await read_json(request))
    return JSONResponse(result, status_code=status)

@app.post('/promote')
async def promote_endpoint(request: Request): return await dispatch(request, handle_promote)
@app.post('/adapt')
async def adapt_endpoint(request: Request): return await dispatch(request, handle_adapt)
@app.post('/quantize')
async def quantize_endpoint(request: Request): return await dispatch(request, handle_quantize)
@app.post('/pipeline')
async def pipeline_endpoint(request: Request): return await dispatch(request, handle_pipeline)
@app.post('/verify-bundle')
async def bundle_endpoint(request: Request): return await dispatch(request, handle_bundle)
