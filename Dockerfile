FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip3 install --no-cache-dir --break-system-packages -r /app/requirements.txt

RUN mkdir -p /models

COPY app/ /app/app/

ENV CHECKPOINT_PATH=/models/decoder_only.pt
ENV TOKENIZER_PATH=/models/tokenizer.pkl
ENV S3_BUCKET=paged-inference-serving
ENV S3_KEY_CHECKPOINT=v1/decoder_only.pt
ENV S3_KEY_TOKENIZER=v1/tokenizer.pkl

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]