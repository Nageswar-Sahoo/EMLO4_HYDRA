import os
import io
import json
import zlib
import socket
import logging

import torch
import requests
import numpy as np
import redis.asyncio as redis
import timm
import torchvision.transforms as transforms
from PIL import Image
from fastapi import FastAPI, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from typing import Dict, Annotated

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - ModelServer - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Mamba Model Server")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "mambaout_base.in1k")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")
HOSTNAME = socket.gethostname()

# Global variables
model = None
device = None
transform = None
categories = []
redis_pool = None

# Model initialization
@app.on_event("startup")
async def initialize():
    global model, device, transform, categories, redis_pool
    model_path = "model.pt"

    logger.info(f"Initializing model server on host {HOSTNAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    try:
        logger.info(f"Loading model: {MODEL_NAME}")
        model = torch.jit.load(model_path)
        model.to(device).eval()
        logger.info("Model successfully loaded and initialized.")
    except (FileNotFoundError, RuntimeError, Exception) as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Define categories
    categories = [
        "beagle", "boxer", "bulldog", "dachshund", "german_Shepherd",
        "Golden_Retriever", "Labrador_Retriever", "Poodle",
        "Rottweiler", "Yorkshire_Terrier"
    ]

    # Setup Redis connection pool
    redis_pool = redis.ConnectionPool(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        db=0,
        decode_responses=True
    )
    logger.info("Model server initialization complete")

# Cleanup on shutdown
@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down model server")
    await redis_pool.aclose()
    logger.info("Redis connection pool closed")

# Redis client dependency
def get_redis():
    return redis.Redis(connection_pool=redis_pool)

# Prediction function
def predict(inp_img: Image) -> Dict[str, float]:
    logger.info("Running inference")
    img_tensor = transform(inp_img.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(img_tensor)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 5)

    return {categories[idx.item()]: float(prob) for prob, idx in zip(top_prob, top_catid)}

# Write to Redis cache
async def write_to_cache(file: bytes, result: Dict[str, float]) -> None:
    cache = get_redis()
    hash_key = str(zlib.adler32(file))
    logger.info(f"Caching result with hash: {hash_key}")
    await cache.set(hash_key, json.dumps(result))

# Inference endpoint
@app.post("/infer")
async def infer(image: Annotated[bytes, File()]) -> Dict[str, float]:
    logger.info("Received inference request")
    img = Image.open(io.BytesIO(image))
    predictions = predict(img)
    await write_to_cache(image, predictions)
    return predictions

# Health check endpoint
@app.get("/health")
async def health_check() -> Dict:
    try:
        redis_client = get_redis()
        redis_connected = await redis_client.ping()
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        redis_connected = False

    return {
        "status": "healthy" if redis_connected else "unhealthy",
        "hostname": HOSTNAME,
        "model": MODEL_NAME,
        "device": str(device) if device else None,
        "redis": {
            "host": REDIS_HOST,
            "port": REDIS_PORT,
            "connected": redis_connected,
        },
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)