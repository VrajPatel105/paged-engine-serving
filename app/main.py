from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import HTTPException
import time
from app.schemas import UserInput

# module level flag
is_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global is_ready
    time.sleep(15)
    is_ready = True
    
    yield

app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message" : "Hey There, Welcome to Paged Inference Serving Engine"}

@app.get("/health")
async def health():
    if not is_ready:
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ready"}


@app.post("/generate")
async def generate(data: UserInput): 
    return {"text": "Hey, this is the model temp output"}
    
