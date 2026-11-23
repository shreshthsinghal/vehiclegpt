# VehicleGPT on-device image: Python app + llama.cpp server. Model and data are mounted, not baked in.
FROM python:3.13-slim AS llama
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp /src \
    && cmake -S /src -B /src/build -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF \
       -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /src/build -j4 --target llama-server

FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY --from=llama /src/build/bin/llama-server /usr/local/bin/llama-server
WORKDIR /app
COPY . /app
ENV VGPT_LLAMA_SERVER=/usr/local/bin/llama-server PYTHONUNBUFFERED=1
EXPOSE 8765
CMD ["python", "-m", "ask.server"]
