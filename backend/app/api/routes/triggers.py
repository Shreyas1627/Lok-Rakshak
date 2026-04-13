from fastapi import APIRouter

router = APIRouter()

@router.post("/dismiss")
async def dismiss_alert():
    """Manual HITL (Human In The Loop) override to dismiss a threat."""
    return {"status": "success", "action": "Threat dismissed. Returning to GREEN."}

@router.post("/escalate")
async def manual_escalate():
    """Admin manually pushes the system to CRITICAL."""
    return {"status": "CRITICAL", "action": "Manual override. Activating NDMA protocols."}