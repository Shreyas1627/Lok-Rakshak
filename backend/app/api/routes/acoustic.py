from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class AcousticData(BaseModel):
    score: int # 0=Normal, 1=Warning, 3=Critical

@router.post("/feed")
async def update_acoustic(data: AcousticData):
    # In a real app, this updates a Redis cache or global state
    return {"status": "success", "message": f"Acoustic risk updated to {data.score}"}