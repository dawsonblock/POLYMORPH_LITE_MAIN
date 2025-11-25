from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List
from retrofitkit.api.security import get_current_user
from retrofitkit.core.app import AppContext
from retrofitkit.core.orchestrator import Orchestrator
from retrofitkit.core.recipe import Recipe
from retrofitkit.compliance.audit import Audit
from retrofitkit.compliance.signatures import Signer, SignatureRequest
from retrofitkit.compliance import approvals as Approvals
from retrofitkit.data.storage import DataStore

router = APIRouter()
ctx = AppContext.load()
orc = Orchestrator(ctx)
store = DataStore(ctx.config.system.data_dir)

class RunRequest(BaseModel):
    recipe_path: str
    simulation: bool = False

class RunBatchRequest(BaseModel):
    recipe_paths: List[str]
    simulation: bool = False

class Approval(BaseModel):
    request_id: int

class PackageReq(BaseModel):
    run_id: str

@router.get("/status")
def status(user=Depends(get_current_user)):
    return {"ok": True, "user": user, "mode": ctx.config.system.mode, "daq": ctx.config.daq.backend, "raman": ctx.config.raman.provider}

@router.get("/runs")
def runs(user=Depends(get_current_user)):
    return store.list_runs(limit=200)

@router.post("/request_run")
def request_run(payload: RunRequest, user=Depends(get_current_user)):
    req_id = Approvals.request(payload.recipe_path, user["email"])
    Audit().record("RUN_REQUEST", user["email"], payload.recipe_path, {"request_id": req_id})
    return {"request_id": req_id, "status": "PENDING", "required_roles": ["Operator","QA"]}

@router.get("/approvals")
def approvals_list(user=Depends(get_current_user)):
    return Approvals.list_pending(limit=200)

@router.post("/approve")
def approve(payload: Approval, user=Depends(get_current_user)):
    Approvals.approve(payload.request_id, user["email"], user["role"])
    Audit().record("RUN_APPROVE", user["email"], f"req:{payload.request_id}", {"role": user["role"]})
    return {"ok": True}

@router.post("/run")
async def run(payload: RunRequest, user=Depends(get_current_user)):
    approved = [a for a in Approvals.list_pending() if a["recipe_path"] == payload.recipe_path and a["status"] == "APPROVED"]
    if not approved:
        raise HTTPException(status_code=403, detail="Two-person approval required")
    recipe = Recipe.from_yaml(payload.recipe_path)
    rid = await orc.run(recipe, user["email"], simulation=payload.simulation)
    return {"run_id": rid}

@router.post("/run_batch")
async def run_batch(payload: RunBatchRequest, user=Depends(get_current_user)):
    run_ids = []
    for rp in payload.recipe_paths:
        recipe = Recipe.from_yaml(rp)
        rid = await orc.run(recipe, user["email"], simulation=payload.simulation)
        run_ids.append(rid)
    return {"run_ids": run_ids}

@router.post("/package_run")
def package_run(payload: PackageReq, user=Depends(get_current_user)):
    path = store.package_run(payload.run_id)
    Audit().record("RUN_PACKAGE", user["email"], payload.run_id, {"path": path})
    return {"package_path": path}
