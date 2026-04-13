# start_server.py
import asyncio
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from pydantic import BaseModel
from app.api.routes import acoustic, triggers, sdk

from app.ml.yolo_inference import VisionEngine
from app.core.risk_engine import RiskEngine
from app.core.state_machine import SystemStateMachine
from app.websocket.manager import manager

app = FastAPI(title="Lok-Rakshak API")

# Register the modular routes
app.include_router(acoustic.router, prefix="/api/acoustic", tags=["Acoustic"])
app.include_router(triggers.router, prefix="/api/triggers", tags=["HITL Control"])
app.include_router(sdk.router, prefix="/api/sdk", tags=["B2G SDK"])

vision_engine = VisionEngine()
risk_engine = RiskEngine()
state_machine = SystemStateMachine()

# Global variables to hold data from other laptops
current_acoustic_score = 0
current_sdk_score = 20

# Define a schema for Laptop 2 to send acoustic data
class AcousticData(BaseModel):
    score: int # 0=Normal, 1=Warning, 3=Critical

@app.post("/api/acoustic")
async def update_acoustic(data: AcousticData):
    global current_acoustic_score
    current_acoustic_score = data.score
    return {"message": "Acoustic score updated"}

@app.post("/api/dismiss")
async def dismiss_alert():
    """Laptop 4 (UI) calls this when admin clicks Dismiss"""
    state_machine.manual_reset()
    return {"message": "System reset to GREEN"}

@app.websocket("/ws/risk")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

async def process_video_feed():
    cap = cv2.VideoCapture(0) # 0 for webcam
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
            
        # 1. Vision ML
        vision_data = vision_engine.process_frame(frame)
        
        # 2. Risk Engine (Combines Vision + Acoustic + SDK)
        predicted_risk = risk_engine.calculate_risk(
            vision_data, 
            acoustic_score=current_acoustic_score, 
            sdk_score=current_sdk_score
        )
        
        # 3. State Machine (Handles the 15s timer and NDMA rules)
        system_response = state_machine.update_state(predicted_risk)
        
        # 4. Package data for the UI Team
        payload = {
            "status": system_response["status"],
            "action": system_response["protocol"]["action"],
            "signage": system_response["protocol"]["signage_message"],
            "density": vision_data["density"]
        }
        
        # 5. Broadcast via WebSockets
        await manager.broadcast(payload)
        await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_video_feed())

if __name__ == "__main__":
    uvicorn.run("start_server:app", host="0.0.0.0", port=8000, reload=True)