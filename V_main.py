from datetime import datetime
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.agent_orch.graph_builder import WorkflowBundle, build_graph
from src.agent_orch.agents.notification import NotificationAgent
from src.agent_orch.agents.action_executor import ScalingAgent
from src.agent_orch.utils.DBConnect import DBConnect
import numpy as np
import pandas as pd
from decimal import Decimal
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel

semaphore = asyncio.Semaphore(10)

# Initialize FastAPI app
app = FastAPI(title="Agent Orchestrator API")

workflows: WorkflowBundle = build_graph()

# Initialize agents
notification_agent = NotificationAgent()
scaling_agent = ScalingAgent()

# In-memory state storage for manual approvals
state_store: Dict[str, Dict[str, Any]] = {}

# Request schemas
class PipelineRequest(BaseModel):
    server_id: str
    resource_type: str

class BulkPipelineRequest(BaseModel):
    server_ids: List[int]
    resource_type: str    

class ApprovalRequest(BaseModel):
    server_id: str
    resource_type: str
    approval: str  # 'yes' or 'no'
    scale_percent: int = 0  # percentage to scale if approved
    run_id: Optional[int] = None

class CreateCRRequest(BaseModel):
    server_id: str
    resource_type: str

class ImplementResizeRequest(BaseModel):
    server_id: str
    resource_type: str

# Root endpoint (health check)
@app.get("/")
def health_check():
    return {"status": "ok", "message": "Agent Orchestrator API is running"}


def convert_numpy_types(obj):
    """Recursively convert NumPy types and other non-JSON types to Python native types."""
    import math
    
    # Handle Decimal from MySQL
    if isinstance(obj, Decimal):
        val = float(obj)
        # Handle inf and nan from Decimal conversion
        if math.isinf(val) or math.isnan(val):
            return None
        return val
    # Handle NumPy scalar types
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        val = float(obj)
        # Replace inf and nan with None for JSON compatibility
        if math.isinf(val) or math.isnan(val):
            return None
        return val
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle standard Python float (check for inf/nan)
    elif isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    # Handle standard Python types
    elif isinstance(obj, dict):
        # Convert both keys AND values to handle numpy types in keys
        return {convert_numpy_types(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (str, int, bool, type(None))):
        return obj
    # Handle pandas objects
    elif isinstance(obj, pd.DataFrame):
        return [convert_numpy_types(record) for record in obj.to_dict(orient="records")]
    elif isinstance(obj, pd.Series):
        return convert_numpy_types(obj.to_dict())
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Timedelta):
        return obj.isoformat()
    # Handle pandas types if present
    elif hasattr(obj, 'item'):  # NumPy scalar with .item() method
        val = obj.item()
        if isinstance(val, float) and (math.isinf(val) or math.isnan(val)):
            return None
        return val
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, set):
        return [convert_numpy_types(item) for item in obj]
    else:
        # Try to convert to string as last resort
        try:
            return str(obj)
        except:
            return obj

def sanitize_for_response(value: Any):
    return convert_numpy_types(value) if value is not None else None

async def process_single_server(server_id: int, resource_type: str):

    async with semaphore:
        state = {
            "server_id": server_id,
            "resource_type": resource_type
        }

        pipeline_state = await workflows.pre_approval.ainvoke(state)

        result = pipeline_state.get("recommendation", {})

        db = DBConnect()

        db.insert("recommendations", {
            "vpu_id": server_id,
            "resource_type": resource_type,
            "decision": result.get("decision"),
            "avg_usage": result.get("avg_usage", 0),
            "max_usage": result.get("max_usage", 0),
            "forecasted_max": result.get("forecasted_max", 0),
            "scale_percent": result.get("scale_percent", 0),
            "itis_ticket_number": result.get("itis_ticket_number"),
            "action_status": result.get("action_status"),
            "remarks": result.get("remarks"),
            "run_id": pipeline_state.get("run_id")
        })

        return {"server_id": server_id, "status": "success"}
    

@app.post("/pipeline") 
async def run_pipeline(req: PipelineRequest):
    state: Dict[str, Any] = {
        "server_id": req.server_id,
        "resource_type": req.resource_type,
    }

    try:
        pipeline_state = await workflows.pre_approval.ainvoke(state)

        result = pipeline_state.get("recommendation", {})

        db = DBConnect()

        db.insert("recommendations", {
            "vpu_id": req.server_id,
            "resource_type": req.resource_type,
            "decision": result.get("decision"),
            "avg_usage": result.get("avg_usage", 0),
            "max_usage": result.get("max_usage", 0),
            "forecasted_max": result.get("forecasted_max", 0),
            "scale_percent": result.get("scale_percent", 0),
            "itis_ticket_number": result.get("itis_ticket_number"),
            "action_status": result.get("action_status"),
            "remarks": result.get("remarks"),
            "run_id": pipeline_state.get("run_id")
        })

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {exc}") from exc

    key = f"{req.server_id}_{req.resource_type}"
    state_store[key] = {
        "state": pipeline_state,
        "status": "awaiting_approval",
        "run_id": pipeline_state.get("run_id"),
        "updated_at": datetime.utcnow().isoformat(),
        "server_id": req.server_id,
        "resource_type": req.resource_type,
    }

    return {
        "run_id": pipeline_state.get("run_id"),
        "recommendation": sanitize_for_response(pipeline_state.get("recommendation")),
        "cost_impact": sanitize_for_response(pipeline_state.get("cost_impact")),
        "anomaly_count": len(pipeline_state.get("anomalies", [])),
        "awaiting_manual_approval": True,
        "forecasts": sanitize_for_response(pipeline_state.get("forecasts")),
        "notification": sanitize_for_response(pipeline_state.get("notification")),
        "plots": sanitize_for_response(pipeline_state.get("plots")),
        "plot_phase1": sanitize_for_response(pipeline_state.get("plot_phase1")),
        "plot_phase2": sanitize_for_response(pipeline_state.get("plot_phase2")),
        "warnings": sanitize_for_response(pipeline_state.get("warnings")),
        "forecast_status": sanitize_for_response(pipeline_state.get("forecast_status")),
        "state_key": key,
    }

@app.post("/pipeline/bulk")
async def run_bulk_pipeline(req: BulkPipelineRequest):

    try:
        tasks = [
            process_single_server(server_id, req.resource_type)
            for server_id in req.server_ids
        ]

        results = await asyncio.gather(*tasks)

        return {
            "message": "Bulk pipeline executed successfully",
            "total_servers": len(req.server_ids),
            "processed": len(results),
            "results": results
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Bulk pipeline failed: {exc}"
        )
    
@app.post("/pipeline/all")
async def run_pipeline_all():

    try:
        db = DBConnect()

        rows = db.select(
            raw_query="""
                SELECT DISTINCT vpu_id
                FROM vm_primary_uri
            """
        )

        server_ids = [row["vpu_id"] for row in rows]

        tasks = [
            process_single_server(server_id, "cpu")
            for server_id in server_ids
        ]

        results = await asyncio.gather(*tasks)

        return {
            "message": "Pipeline executed for all servers",
            "total_servers": len(server_ids),
            "processed": len(results)
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))    

@app.get("/state")
def get_state(server_id: str, resource_type: str):
    key = f"{server_id}_{resource_type}"
    if key not in state_store:
        raise HTTPException(status_code=404, detail="State not found for this server/resource")

    record = state_store[key]
    sanitized_state = sanitize_for_response(record.get("state"))
    return {
        "server_id": record.get("server_id", server_id),
        "resource_type": record.get("resource_type", resource_type),
        "status": record.get("status"),
        "run_id": record.get("run_id"),
        "updated_at": record.get("updated_at"),
        "state": sanitized_state,
    }


@app.get("/servers")
def list_servers():
    db = None
    try:
        db = DBConnect()
        rows = db.select(
            raw_query=(
                "SELECT DISTINCT a.vm_id, b.class AS server_name "
                "FROM vm_primary_uri a "
                "LEFT JOIN vm_details b ON a.vm_id = b.vm_id "
                "ORDER BY a.vm_id"
            )
        )
    except Exception as exc:  # pragma: no cover - database dependant
        raise HTTPException(status_code=500, detail=f"Failed to load servers: {exc}") from exc
    finally:
        try:
            db.close()
        except Exception:
            pass

    servers = []
    for row in rows:
        vm_id = row.get("vm_id")
        if vm_id is not None:
            servers.append({
                "vm_id": vm_id,
                "server_name": row.get("server_name") or str(vm_id),
            })
    return {"servers": servers}


@app.get("/servers/{server_id}/resources")
def list_server_resources(server_id: str):
    db = None
    try:
        db = DBConnect()
        rows = db.select(
            raw_query=(
                "SELECT DISTINCT resource_type "
                "FROM vm_primary_uri WHERE vm_id = %s ORDER BY resource_type"
            ),
            params=(server_id,),
        )
    except Exception as exc:  # pragma: no cover - database dependant
        raise HTTPException(status_code=500, detail=f"Failed to load resources: {exc}") from exc
    finally:
        try:
            db.close()
        except Exception:
            pass

    resources = [row.get("resource_type") for row in rows if row.get("resource_type") is not None]
    return {"server_id": server_id, "resources": resources}


# Manual approval endpoint
@app.post("/approve_scaling/")
async def approve_scaling(request: ApprovalRequest):
    key = f"{request.server_id}_{request.resource_type}"
    
    # Get existing state
    if key not in state_store:
        raise HTTPException(status_code=404, detail="State not found for this server/resource")

    stored_record = state_store[key]
    if request.run_id is not None and stored_record.get("run_id") != request.run_id:
        raise HTTPException(
            status_code=409,
            detail="Run ID mismatch. Please re-run the pipeline or provide the latest run_id.",
        )

    state = stored_record["state"]
    recommendation = state.get("recommendation")
    if not recommendation:
        raise HTTPException(status_code=400, detail="Recommendation not available; rerun pipeline first.")

    approval_value = request.approval.strip().lower()
    if approval_value not in {"yes", "no"}:
        raise HTTPException(status_code=400, detail="Approval must be either 'yes' or 'no'.")

    manual_proceed = approval_value == "yes"
    state["manual_proceed"] = manual_proceed

    if manual_proceed:
        if request.scale_percent:
            recommendation["scale_percent"] = request.scale_percent
    else:
        recommendation.setdefault("scale_percent", 0)

    state["recommendation"] = recommendation

    # Optional: log/manual approve in notification agent
    state = notification_agent.manual_approve(
        state=state,
        server_id=request.server_id,
        resource_type=request.resource_type,
        approval=request.approval,
    )

    try:
        updated_state = await workflows.post_approval.ainvoke(state)
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"Post-approval workflow failed: {exc}") from exc

    stored_record["state"] = updated_state
    stored_record["status"] = "approved" if manual_proceed else "rejected"
    stored_record["updated_at"] = datetime.utcnow().isoformat()

    # Extract CR fields — LangGraph StateGraph(dict) may drop non-original keys,
    # so fall back to parsing the feedback string for the CR number.
    cr_created = updated_state.get("cr_created", False)
    change_sys_id = updated_state.get("change_sys_id")
    change_number = updated_state.get("change_number")

    if not cr_created:
        import re
        feedback_str = updated_state.get("feedback") or ""
        cr_match = re.search(r"Change Request (CHG\d+) created", feedback_str)
        if cr_match:
            cr_created = True
            change_number = change_number or cr_match.group(1)

    # Persist CR info in the stored state so /check_cr_approval and
    # /implement_resize can find it even if LangGraph didn't propagate it.
    if cr_created:
        stored_record["state"]["cr_created"] = True
        if change_sys_id:
            stored_record["state"]["change_sys_id"] = change_sys_id
        if change_number:
            stored_record["state"]["change_number"] = change_number

    state_store[key] = stored_record

    response_payload = {
        "run_id": stored_record.get("run_id"),
        "manual_proceed": manual_proceed,
        "scaling_executed": updated_state.get("scaling_executed", False),
        "feedback": updated_state.get("feedback"),
        "scaling_result": sanitize_for_response(updated_state.get("scaling_result")),
        "dashboard": updated_state.get("dashboard"),
        "cost_impact": sanitize_for_response(updated_state.get("cost_impact")),
        "recommendation": sanitize_for_response(updated_state.get("recommendation")),
        "state_status": stored_record["status"],
        "cr_created": cr_created,
        "change_sys_id": change_sys_id,
        "change_number": change_number,
    }

    return response_payload

@app.get("/recommendations")
def get_all_recommendations():
    """Get LATEST recommendation for each vpu_id"""
    db = None
    try:
        db = DBConnect()
        rows = db.select(
            raw_query="""
            SELECT
            r.id AS r_id,
            v.resource_type,
            d.class as 'Server Name',
            r.decision,
            r.avg_usage,
            r.max_usage,
            r.forecasted_max,
            r.scale_percent,
            r.itis_ticket_number,
            r.action_status,
            r.remarks
            FROM recommendations r
            JOIN vm_primary_uri v
                ON r.vpu_id = v.vpu_id
            JOIN vm_details d
                ON v.vm_id = d.vm_id
            ORDER BY r.created_at DESC
            """
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load recommendations: {exc}")
    finally:
        try:
            db.close()
        except:
            pass

    recommendations = [
    {
        "r_id": row.get("r_id"),
        "Server Name": row.get("Server Name"),
        "Resource Type": row.get("resource_type"),
        "decision": row.get("decision"),
        "avg_usage": float(row.get("avg_usage", 0)),
        "max_usage": float(row.get("max_usage", 0)),
        "forecasted_max": float(row.get("forecasted_max", 0)),
        "scale_percent": float(row.get("scale_percent", 0)),
        "itis_ticket_number": row.get("itis_ticket_number"),
        "action_status": row.get("action_status"),
        "remarks": row.get("remarks"),
    }
    for row in rows
    ]
    return {"recommendations": recommendations, "total": len(recommendations)}


# ------------------------------------------------------------------
# Change Request Workflow Endpoints (3-step UI flow)
# ------------------------------------------------------------------

@app.post("/create_cr")
async def create_change_request(req: CreateCRRequest):
    """Step 1: Create a ServiceNow Change Request and submit for approval."""
    key = f"{req.server_id}_{req.resource_type}"

    if key not in state_store:
        raise HTTPException(
            status_code=404,
            detail="No pipeline state found. Run the pipeline and submit approval first.",
        )

    stored_record = state_store[key]
    state = stored_record["state"]

    recommendation = state.get("recommendation")
    if not recommendation:
        raise HTTPException(status_code=400, detail="No recommendation in state. Rerun the pipeline.")

    if not state.get("manual_proceed"):
        raise HTTPException(status_code=400, detail="Approval has not been granted yet. Submit approval first.")

    decision = recommendation.get("decision", "")
    if decision == "no_change":
        raise HTTPException(status_code=400, detail="Decision is 'no_change'. No CR needed.")

    # Call ScalingAgent.create_change_request
    try:
        updated_state = scaling_agent.create_change_request(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"CR creation failed: {exc}") from exc

    stored_record["state"] = updated_state
    stored_record["updated_at"] = datetime.utcnow().isoformat()
    state_store[key] = stored_record

    return {
        "cr_created": updated_state.get("cr_created", False),
        "change_sys_id": updated_state.get("change_sys_id"),
        "change_number": updated_state.get("change_number"),
        "feedback": updated_state.get("feedback"),
    }


@app.get("/check_cr_approval")
def check_cr_approval(server_id: str, resource_type: str):
    """Step 2: Check whether the CR has been approved in ServiceNow."""
    key = f"{server_id}_{resource_type}"

    if key not in state_store:
        raise HTTPException(status_code=404, detail="No pipeline state found.")

    stored_record = state_store[key]
    state = stored_record["state"]

    if not state.get("change_sys_id"):
        raise HTTPException(status_code=400, detail="No Change Request exists. Create one first.")

    try:
        updated_state = scaling_agent.check_cr_approval(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Approval check failed: {exc}") from exc

    stored_record["state"] = updated_state
    stored_record["updated_at"] = datetime.utcnow().isoformat()
    state_store[key] = stored_record

    return {
        "cr_approved": updated_state.get("cr_approved", False),
        "cr_cancelled": updated_state.get("cr_cancelled", False),
        "cr_approval_message": updated_state.get("cr_approval_message", ""),
        "change_number": updated_state.get("change_number"),
    }


@app.post("/implement_resize")
async def implement_resize(req: ImplementResizeRequest):
    """Step 3: Move CR to Implement, perform Azure VM resize, close/update CR."""
    key = f"{req.server_id}_{req.resource_type}"

    if key not in state_store:
        raise HTTPException(status_code=404, detail="No pipeline state found.")

    stored_record = state_store[key]
    state = stored_record["state"]

    if not state.get("change_sys_id"):
        raise HTTPException(status_code=400, detail="No Change Request found. Create one first.")

    if not state.get("cr_approved"):
        raise HTTPException(status_code=400, detail="CR is not yet approved. Check approval first.")

    try:
        updated_state = scaling_agent.implement_resize(state)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Implement resize failed: {exc}") from exc

    stored_record["state"] = updated_state
    stored_record["status"] = "implemented"
    stored_record["updated_at"] = datetime.utcnow().isoformat()
    state_store[key] = stored_record

    return {
        "scaling_executed": updated_state.get("scaling_executed", False),
        "feedback": updated_state.get("feedback"),
        "scaling_result": sanitize_for_response(updated_state.get("scaling_result")),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run('main:app', host="0.0.0.0", port=8000, log_level="info")
