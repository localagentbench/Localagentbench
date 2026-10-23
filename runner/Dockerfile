FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir requests==2.32.3
COPY lab ./lab
COPY results.schema.json ./
ENTRYPOINT ["python", "-m", "lab"]
# docker run --rm --network host -v $PWD/results:/app/results lab --base-url http://localhost:11434/v1 --model qwen2.5:1.5b
