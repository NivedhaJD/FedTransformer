import asyncio
import json
import threading
from typing import List

import uvicorn
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from federated_learning.coordinator import FederatedCoordinator
from network.client import build_clients

app = FastAPI(title="FedTransformer 6G-ISAC API")

# Serve frontend directory
app.mount("/src", StaticFiles(directory="frontend/src"), name="src")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"WebSocket send error: {e}")

manager = ConnectionManager()
main_loop = None

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()

def emit_step(data: dict):
    """Callback fired by FederatedCoordinator in the PyTorch background thread."""
    if main_loop and main_loop.is_running():
        asyncio.run_coroutine_threadsafe(manager.broadcast(json.dumps(data)), main_loop)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            cmd = json.loads(data)
            
            if cmd.get("action") == "start":
                print("[API] Starting Federated Learning PyTorch thread...")
                # Start training in background thread to avoid blocking the event loop
                def run_training():
                    device = torch.device("cpu")
                    # Lighter load for UI responsiveness
                    clients = build_clients(num_clients=10, samples_per_client=200, local_epochs=1, device=device)
                    server = FederatedCoordinator(
                        num_clients=10, 
                        clients_per_round=4, 
                        num_rounds=5, 
                        device=device,
                        on_step_callback=emit_step
                    )
                    server.run(clients)
                    emit_step({"step": "done", "action": "training_complete"})
                
                threading.Thread(target=run_training, daemon=True).start()
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    print("="*60)
    print("  Starting FedTransformer API + UI Server")
    print("  Access the dashboard at: http://localhost:8000")
    print("="*60)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, log_level="warning")
