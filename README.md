# paged-engine-serving

Production serving layer for [paged-inference-engine](https://github.com/VrajPatel105/paged-inference-engine), a mini-vLLM style inference engine I built from scratch (paged KV cache, continuous batching, FlashAttention-2 Triton kernel, INT8 KV quantization).

The engine stays a library. This repo holds only the API, container, CI/CD, and deploy config.

## API

| Endpoint | Description |
|---|---|
| `POST /generate` | Takes a prompt and optional `max_tokens`, returns generated text. Blocks until that request's sequence finishes. |
| `GET /health` | Returns 200 only after the model is loaded and the Triton kernel has been warmed up. 503 otherwise. |
| `GET /metrics` | Free/total KV blocks, waiting and running sequences, total requests served. |

The engine loop runs on a background thread started at app startup. Requests are submitted to a queue and the scheduler batches them together, so concurrent requests are served in the same forward pass.

## Stack

FastAPI, Pydantic, Docker, GitHub Actions, AWS (ECR, ECS, S3, IAM), pytest.

## How it fits together

- Engine is pip installed from a git tag, so the serving repo never copies engine code.
- Model checkpoint and tokenizer live in S3 and are downloaded at container startup, not baked into the image. Credentials come from the ECS task role, so nothing sensitive is in the image.
- CI runs the API tests on every push. On main, it builds the image and pushes it to ECR tagged with the commit SHA.
- Tests mock the engine so they run on CPU runners, since GitHub's free runners have no GPU.

## Startup cost

Cold start is about 4 seconds locally: roughly 1.4s to load the checkpoint and 2.8s for warmup, which includes the Triton JIT compile. In a fresh container the compile cache is empty, so the real number is higher. This is why `/health` gates on warmup completing rather than on the process starting.

## Image size

Started at 26GB on the CUDA `devel` base image. Switched to `runtime` and it dropped to 17.4GB, with the Triton kernel still compiling fine at runtime.

## Deployment status

ECS cluster and task definitions are in `deploy/`. Not run on a live GPU instance: my account's G-instance vCPU quota is 0 and I did not pursue the increase for a portfolio project. Everything up to and including the ECR push is working.

## Running locally

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Needs `CHECKPOINT_PATH`, `TOKENIZER_PATH`, `S3_BUCKET`, `S3_KEY_CHECKPOINT`, `S3_KEY_TOKENIZER` in the environment or a `.env` file, plus AWS credentials for the S3 download.