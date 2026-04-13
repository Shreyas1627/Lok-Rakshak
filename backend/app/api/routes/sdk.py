from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class SDKData(BaseModel):
    user_id: str
    reported_crowd_level: int # 1 to 100

@router.post("/telemetry")
async def receive_sdk_data(data: SDKData):
    """Receives live GPS and crowd telemetry from the Yatri app SDK."""
    return {"status": "logged", "user": data.user_id}