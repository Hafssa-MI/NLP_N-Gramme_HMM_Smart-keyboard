import json
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.engine import Predictor, UserCache

ROOT = Path(__file__).resolve().parent.parent
bundle = joblib.load(ROOT / "models" / "engine.joblib")
engine = Predictor(bundle["kn"], bundle["hmm"], bundle["vocab"], bundle["beta"])
USERS: "OrderedDict[str, UserCache]" = OrderedDict()     # in-memory only; nothing is written to disk

app = FastAPI(title="Smart Keyboard")


def get_user(uid: Optional[str], create: bool = False):
    if not uid:
        return None
    c = USERS.get(uid)
    if c is None and create:
        c = USERS[uid] = UserCache()
        if len(USERS) > 500:
            USERS.popitem(last=False)
    if c is not None:
        USERS.move_to_end(uid)
    return c


class SuggestIn(BaseModel):
    text: str = Field("", max_length=2000)
    k: int = Field(3, ge=1, le=8)
    user_id: Optional[str] = Field(None, max_length=64)
    use_hmm: bool = True
    personalize: bool = True


class CompleteIn(BaseModel):
    text: str = Field("", max_length=2000)
    length: int = Field(4, ge=1, le=8)
    beam: int = Field(3, ge=1, le=5)
    use_hmm: bool = True


class LearnIn(BaseModel):
    text: str = Field(..., max_length=20000)
    user_id: str = Field(..., max_length=64)


class ForgetIn(BaseModel):
    user_id: str = Field(..., max_length=64)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/info")
def info():
    return {"vocab": engine.vocab.V, "beta": engine.beta, "tags": engine.hmm.tags, **bundle["meta"]}


@app.post("/api/suggest")
def suggest(body: SuggestIn):
    user = get_user(body.user_id) if body.personalize else None
    return engine.suggest(body.text, body.k, user=user, use_hmm=body.use_hmm)


@app.post("/api/complete")
def complete(body: CompleteIn):
    return {"completions": engine.complete(body.text, body.length, body.beam, body.use_hmm)}


@app.post("/api/learn")
def learn(body: LearnIn):
    return get_user(body.user_id, create=True).learn(body.text, engine.vocab)


@app.post("/api/forget")
def forget(body: ForgetIn):
    USERS.pop(body.user_id, None)
    return {"ok": True}


@app.get("/api/metrics")
def metrics():
    path = ROOT / "results" / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "metrics.json not found")
    return json.loads(path.read_text())


app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="web")
