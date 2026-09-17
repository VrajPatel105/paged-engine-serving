from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi import HTTPException
from app.schemas import UserInput
import queue
from core.model_runner import Engine
import pickle
import threading
from transformer.load_checkpoint import load_trained_weights
import time
import os
from dotenv import load_dotenv
load_dotenv()
TIMEOUT=60

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH")

engine = None
engine_thread = None
is_ready = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global engine, engine_thread, is_ready

    global_start_time = time.time()

    model = load_trained_weights(CHECKPOINT_PATH)

    with open(TOKENIZER_PATH, 'rb') as f:
        tok = pickle.load(f)

    engine = Engine(model, tok)
    engine_thread = threading.Thread(target=engine.run, daemon=True)
    engine_thread.start()

    #warmup
    warmup_sentence =  "This is a warmup text to turn on the engineeeeeeeeee (vroommmmmmmm)"
    warmup_seq_id = engine.submit(tok.encode_sentence(warmup_sentence,  add_sos=True, add_eos=False), 20) # submitting the list of ints (encoded from tokenizer)
    start_time = time.time()
    warmup_ok = 0  # 0 : success, 1 : failure
    while True:
        entry = engine.results.get(warmup_seq_id)
        if entry is not None and entry[1]:
            engine.results.pop(warmup_seq_id, None)
            break

        elapsed_time = time.time() - start_time
        if elapsed_time > 30:
            print("broke the loop since it took more than 30 seconds for warmup")
            warmup_ok = 1
            break
        time.sleep(0.1)

    if warmup_ok == 0:
        print("warmup done \n")
        print("total time taken: ", time.time() - start_time)
        print("global start time", time.time() - global_start_time)
        is_ready = True

    yield

    # for stopping, get the stop method in engine
    engine.stop()
    engine_thread.join()

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
def generate(data: UserInput):

    output_id = engine.submit(engine.tok.encode_sentence(data.prompt, add_sos=True, add_eos=False), data.max_tokens)
    start = time.time()
    try:
        while True:
            entry = engine.results.get(output_id)
            if entry is not None and entry[1]:
                return {"text": entry[0]}
            if time.time() - start > TIMEOUT:
                raise HTTPException(status_code=504, detail="generation timed out")
            time.sleep(0.05)
    finally:
        print("length of engine.results before: ", len(engine.results))
        engine.results.pop(output_id, None)
        print("length of engine.results after: ", len(engine.results))
        
    
